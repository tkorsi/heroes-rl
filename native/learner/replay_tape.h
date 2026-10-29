// A bounded executor for the retained, first-order CPU text backward graph.
// Builtin ATen nodes and InputBuffer retain all numerical operations. The tape
// removes only repeated Engine scheduling and thread-local-state restoration.
#pragma once

#include <ATen/FuncTorchTLS.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/autocast_mode.h>
#include <ATen/record_function.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/generated/Functions.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace native_text {
class BuiltinReplayTape {
    using Node = torch::autograd::Node;
    using Edge = torch::autograd::Edge;
    using Variables = torch::autograd::variable_list;
    using PostHook = torch::autograd::FunctionPostHook;
    using InputBuffer = torch::autograd::InputBuffer;
    static constexpr size_t maximum_nodes = 65536;
    static constexpr size_t no_target = static_cast<size_t>(-1);

    struct Route {
        size_t target;
        uint32_t input;
    };
    struct Entry {
        std::shared_ptr<Node> node;
        std::vector<Route> routes;
        std::vector<const PostHook*> hooks;
        uint64_t sequence;
        uint32_t inputs;
        size_t alias = no_target;
    };
    Edge root;
    size_t alias_count = 0;
    std::vector<Entry> entries;
    std::vector<size_t> schedule;

    static bool reject(std::string* reason, const char* message) {
        if (reason) *reason = message;
        return false;
    }
    static bool ordinary_tensor(const at::Tensor& tensor) {
        return tensor.defined() && tensor.device().is_cpu() && tensor.scalar_type() == at::kFloat &&
            tensor.layout() == at::kStrided && !tensor.is_nested() && !tensor.is_inference() &&
            !tensor._fw_grad(0).defined() &&
            !tensor._is_zerotensor() && !at::isTensorSubclassLike(tensor);
    }
    static bool ordinary_context(std::string* reason) {
        if (at::get_num_threads() != 1)
            return reject(reason, "The tape requires one ATen intra-op thread");
        if (torch::autograd::get_current_graph_task_id() != -1)
            return reject(reason, "An enclosing autograd GraphTask requires the ordinary Engine");
        if (at::hasCallbacks()) return reject(reason, "RecordFunction callbacks require the ordinary Engine");
        if (torch::autograd::AnomalyMode::is_enabled())
            return reject(reason, "Anomaly detection requires the ordinary Engine");
        if (!at::SavedTensorDefaultHooks::get_tls_state().stack.empty())
            return reject(reason, "Saved-tensor hooks require the ordinary Engine");
        if (c10::InferenceMode::is_enabled() || at::autocast::is_autocast_enabled(at::kCPU))
            return reject(reason, "Inference or autocast mode requires the ordinary Engine");
        if (c10::impl::TorchDispatchModeTLS::any_modes_set() ||
            c10::impl::PythonDispatcherTLS::get_state() ||
            at::impl::PythonTorchFunctionTLS::stack_len() || at::functorch::functorchTLSAccessor())
            return reject(reason, "Custom dispatch state requires the ordinary Engine");
        return true;
    }
    static bool builtin_without_saved_hooks(const Node& node) {
        using namespace torch::autograd::generated;
        const auto& type = typeid(node);
        // Exact dynamic types prevent a custom subclass from entering the tape.
        if (type == typeid(AddmmBackward0)) {
            const auto& value = static_cast<const AddmmBackward0&>(node);
            return !value.mat1_.has_hooks() && !value.mat2_.has_hooks();
        }
        if (type == typeid(MulBackward0)) {
            const auto& value = static_cast<const MulBackward0&>(node);
            return !value.self_.has_hooks() && !value.other_.has_hooks();
        }
        if (type == typeid(SigmoidBackward0))
            return !static_cast<const SigmoidBackward0&>(node).result_.has_hooks();
        if (type == typeid(TanhBackward0))
            return !static_cast<const TanhBackward0&>(node).result_.has_hooks();
        if (type == typeid(EmbeddingBackward0)) {
            const auto& value = static_cast<const EmbeddingBackward0&>(node);
            return !value.indices_.has_hooks() && !value.sparse;
        }
        return type == typeid(AddBackward0) || type == typeid(SubBackward0) ||
            type == typeid(ViewBackward0) || type == typeid(UnsafeViewBackward0) ||
            type == typeid(SelectBackward0) || type == typeid(TBackward0) ||
            type == typeid(TransposeBackward0) || type == typeid(UnsqueezeBackward0) ||
            type == typeid(UnbindBackward0) || type == typeid(UnsafeSplitBackward0) ||
            type == typeid(StackBackward0);
    }
    static bool ordinary_metadata(Node& node) {
        if (!node.device().is_cpu() || node.stream().has_value()) return false;
        for (size_t i = 0; i < node.num_inputs(); ++i) {
            const auto& metadata = node.input_metadata(i);
            if (metadata.was_default_constructed() || !metadata.device().is_cpu() ||
                metadata.layout() != at::kStrided || metadata.is_tensor_subclass() ||
                metadata.is_nested_tensor() || metadata.dtype() != caffe2::TypeMeta::Make<float>()) return false;
        }
        return true;
    }
    static bool has_external_pre_hooks(Node& node) {
        return !node.pre_hooks().empty() || !node.tensor_pre_hooks().empty() || !node.retains_grad_hooks().empty();
    }

public:
    // Only the caller's own contribution hooks may be accepted. All other hooks
    // force fallback before any backward node is executed.
    static std::shared_ptr<BuiltinReplayTape> compile(
        const at::Tensor& output, const Variables& aliases,
        const std::unordered_set<const PostHook*>& allowed_post_hooks = {}, std::string* reason = nullptr) {
        if (!ordinary_context(reason)) return nullptr;
        if (!ordinary_tensor(output) || !output.grad_fn() || aliases.empty()) {
            reject(reason, "The tape requires a differentiable dense CPU float output and leaf aliases");
            return nullptr;
        }
        auto tape = std::shared_ptr<BuiltinReplayTape>(new BuiltinReplayTape);
        tape->root = torch::autograd::impl::gradient_edge(output);
        tape->alias_count = aliases.size();
        std::unordered_map<Node*, size_t> alias_nodes;
        for (size_t i = 0; i < aliases.size(); ++i) {
            const auto& alias = aliases[i];
            if (!ordinary_tensor(alias) || !alias.is_leaf() || !alias.requires_grad()) {
                reject(reason, "The tape requires ordinary differentiable leaf aliases");
                return nullptr;
            }
            const auto edge = torch::autograd::impl::gradient_edge(alias);
            const auto* alias_node = edge.function.get();
            if (!alias_node || typeid(*alias_node) != typeid(torch::autograd::AccumulateGrad) ||
                edge.input_nr != 0 || !alias_nodes.emplace(edge.function.get(), i).second) {
                reject(reason, "The tape requires distinct alias accumulation nodes");
                return nullptr;
            }
        }
        std::vector<std::shared_ptr<Node>> nodes{tape->root.function};
        std::unordered_map<Node*, size_t> indices{{tape->root.function.get(), 0}};
        std::unordered_set<uint64_t> sequences;
        size_t reached_aliases = 0;
        for (size_t index = 0; index < nodes.size(); ++index) {
            // Copy the shared pointer: appending children may reallocate nodes.
            const auto node = nodes[index];
            const auto alias = alias_nodes.find(node.get());
            const bool terminal = alias != alias_nodes.end();
            if (nodes.size() > maximum_nodes || has_external_pre_hooks(*node) || !ordinary_metadata(*node) ||
                (!terminal && !builtin_without_saved_hooks(*node))) {
                reject(reason, "Unsupported node, metadata, hooks or graph size");
                return nullptr;
            }
            Entry entry{node, {}, {}, node->sequence_nr(), node->num_inputs(), terminal ? alias->second : no_target};
            if (terminal) {
                ++reached_aliases;
                if (!node->next_edges().empty() || !node->post_hooks().empty()) {
                    reject(reason, "Alias terminals must have no outgoing edges or hooks");
                    return nullptr;
                }
            } else if (!sequences.insert(node->sequence_nr()).second) {
                reject(reason, "Equal nonterminal sequence numbers require the ordinary Engine");
                return nullptr;
            }
            for (const auto& hook : node->post_hooks()) {
                if (!allowed_post_hooks.count(hook.get())) {
                    reject(reason, "An external post-hook requires the ordinary Engine");
                    return nullptr;
                }
                entry.hooks.push_back(hook.get());
            }
            for (const auto& edge : node->next_edges()) {
                if (!edge.is_valid()) { entry.routes.push_back({no_target, edge.input_nr}); continue; }
                if (edge.input_nr >= edge.function->num_inputs()) {
                    reject(reason, "An edge references an invalid input slot");
                    return nullptr;
                }
                const auto [it, inserted] = indices.emplace(edge.function.get(), nodes.size());
                if (inserted) nodes.push_back(edge.function);
                entry.routes.push_back({it->second, edge.input_nr});
            }
            tape->entries.push_back(std::move(entry));
        }
        if (reached_aliases != aliases.size()) {
            reject(reason, "Every alias must be reachable from the output");
            return nullptr;
        }
        // Match ReadyQueue's sequence ordering and edge insertion ordering.
        // All nodes are CPU nodes from one non-reentrant graph task.
        std::vector<size_t> dependencies(nodes.size(), 0);
        for (const auto& entry : tape->entries)
            for (const auto& route : entry.routes) if (route.target != no_target) ++dependencies[route.target];
        const auto compare = [&](size_t left, size_t right) {
            return tape->entries[left].sequence < tape->entries[right].sequence;
        };
        std::priority_queue<size_t, std::vector<size_t>, decltype(compare)> ready(compare);
        if (dependencies[0] != 0) {
            reject(reason, "The retained backward graph has a cycle");
            return nullptr;
        }
        ready.push(0);
        while (!ready.empty()) {
            const auto index = ready.top(); ready.pop();
            tape->schedule.push_back(index);
            for (const auto& route : tape->entries[index].routes)
                if (route.target != no_target && --dependencies[route.target] == 0) ready.push(route.target);
        }
        if (tape->schedule.size() != tape->entries.size()) {
            reject(reason, "The retained backward graph is not a rooted DAG");
            return nullptr;
        }
        // This proof makes an absent GraphTask equivalent to grad(..., aliases):
        // every valid edge is needed, with alias nodes used only for capture.
        std::vector<bool> reaches_alias(nodes.size(), false);
        for (auto it = tape->schedule.rbegin(); it != tape->schedule.rend(); ++it) {
            const auto& entry = tape->entries[*it];
            bool reaches = entry.alias != no_target;
            for (const auto& route : entry.routes)
                if (route.target != no_target) reaches = reaches || reaches_alias[route.target];
            if (!reaches) {
                reject(reason, "A backward branch does not reach a permitted alias");
                return nullptr;
            }
            reaches_alias[*it] = true;
        }
        if (reason) reason->clear();
        return tape;
    }

    bool supported_now(std::string* reason = nullptr) const {
        if (!ordinary_context(reason)) return false;
        // These retained graphs are private. Still reject changed hooks or
        // connectivity before execution rather than silently using stale plans.
        for (const auto& entry : entries) {
            const auto& node = *entry.node;
            if (node.sequence_nr() != entry.sequence || node.num_inputs() != entry.inputs ||
                node.next_edges().size() != entry.routes.size() || has_external_pre_hooks(*entry.node) ||
                node.post_hooks().size() != entry.hooks.size() ||
                (entry.alias == no_target && !builtin_without_saved_hooks(node)))
                return reject(reason, "The retained backward graph changed after compilation");
            for (size_t i = 0; i < entry.hooks.size(); ++i)
                if (node.post_hooks()[i].get() != entry.hooks[i])
                    return reject(reason, "The retained backward hooks changed after compilation");
            for (size_t i = 0; i < entry.routes.size(); ++i) {
                const auto& edge = node.next_edge(i);
                const auto& route = entry.routes[i];
                const auto expected = route.target == no_target ? nullptr : entries[route.target].node.get();
                if (edge.function.get() != expected || edge.input_nr != route.input)
                    return reject(reason, "The retained backward edges changed after compilation");
            }
        }
        if (reason) reason->clear();
        return true;
    }

    // False is a pre-execution fallback decision. Once a node starts, all
    // exceptions propagate; partially captured contributions are never replayed.
    bool run_if_supported(const at::Tensor& gradient, Variables* terminal_gradients = nullptr) const {
        if (!ordinary_tensor(gradient) || !supported_now()) return false;
        at::AutoGradMode no_higher_order_graph(false);
        Variables seeds{gradient};
        torch::autograd::validate_outputs({root}, seeds, [](const std::string& message) { return message; });
        std::vector<InputBuffer> buffers;
        buffers.reserve(entries.size());
        // CPU InputBuffer::add only uses buffer. This constructor avoids three
        // unused accelerator stream/event vectors per node, without changing add.
        for (const auto& entry : entries) buffers.emplace_back(Variables(entry.inputs));
        auto root_input = seeds[0];
        buffers[0].add(root.input_nr, std::move(root_input), std::nullopt, std::nullopt);
        Variables captured(alias_count);
        for (const auto index : schedule) {
            const auto& entry = entries[index];
            Variables outputs;
            {
                // InputBuffer::variables is hidden in this libtorch build. Its
                // complete implementation moves the public buffer member.
                auto inputs = std::move(buffers[index].buffer);
                if (entry.alias != no_target) {
                    captured[entry.alias] = inputs.at(0);
                    continue;
                }
                if (entry.hooks.empty()) outputs = (*entry.node)(std::move(inputs));
                else {
                    // Engine::call_function preserves inputs while post-hooks run.
                    auto inputs_copy = inputs;
                    outputs = (*entry.node)(std::move(inputs_copy));
                }
                torch::autograd::validate_outputs(entry.node->next_edges(), outputs,
                    [&](const std::string& message) { return "Function " + entry.node->name() + " returned an " + message; });
                for (const auto& hook : entry.node->post_hooks()) outputs = (*hook)(outputs, inputs);
            }
            // Engine destroys call_function's inputs before routing outputs.
            // This lifetime affects InputBuffer's reference-count-based reuse.
            TORCH_CHECK(outputs.size() == entry.routes.size(), "Native replay post-hook changed output count");
            for (size_t i = 0; i < entry.routes.size(); ++i) {
                const auto& route = entry.routes[i];
                if (route.target != no_target)
                    buffers[route.target].add(route.input, std::move(outputs[i]), std::nullopt, std::nullopt);
            }
        }
        if (terminal_gradients) *terminal_gradients = std::move(captured);
        return true;
    }
    size_t node_count() const { return entries.size(); }
};
} // namespace native_text
