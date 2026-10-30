// Native ATen execution of the existing contextual recurrent policy and PPO loss.
// Parameter tensors remain owned by the Python model and its existing Adam optimizer.
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/SequenceNumber.h>
#include <ATen/ThreadLocalState.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using at::Tensor;
using Cache = std::unordered_map<int64_t, Tensor>;

#include "text_replay.h"

namespace {
void require(bool condition, const std::string& message) {
    TORCH_CHECK(condition, message);
}
Tensor recipe_tensor(const py::handle& object, at::ScalarType dtype, int64_t rank,
                     const std::string& label) {
    auto tensor = py::cast<Tensor>(object);
    require(tensor.device().is_cpu() && tensor.scalar_type() == dtype && tensor.dim() == rank,
            "Invalid native recipe tensor: " + label);
    require(!tensor.requires_grad(), "Recipe inputs must be detached constants: " + label);
    if (tensor.is_floating_point())
        require(at::isfinite(tensor).all().item<bool>(), "Nonfinite recipe tensor: " + label);
    return tensor.detach().clone();
}
struct Node {
    std::string kind;
    std::string text;
    std::vector<double> number;
    int64_t tag = -1, canonical = -1;
    std::vector<int64_t> children;
};
struct Map {
    int64_t width = 0, height = 0, own = -1;
    Tensor terrain, road, flags, attributes, indices;
};
struct Conditional {
    std::string kind;
    int64_t selected = -1, root = -1;
    std::vector<int64_t> roots;
    Tensor positions, indices, numeric, offsets;
};
struct Frame {
    int64_t root = -1;
    std::optional<Map> map;
    std::vector<Conditional> conditionals;
    bool decision = false;
};
}

class NativeHistory {
public:
    std::vector<Node> nodes;
    std::vector<Frame> frames;
    NativeHistory(const py::list& raw_nodes, const py::list& raw_frames) {
        require(raw_nodes.size() < 10000000 && raw_frames.size() < 1000000,
                "Native history exceeds bounded recipe size");
        for (const auto& item : raw_nodes) {
            auto row = py::cast<py::tuple>(item);
            require(row.size() == 3, "A native tree node must have three fields");
            Node node;
            node.kind = py::cast<std::string>(row[0]);
            node.children = py::cast<std::vector<int64_t>>(row[2]);
            if (node.kind == "text") {
                node.text = py::cast<std::string>(row[1]);
                require(node.children.empty(), "Text nodes cannot have children");
            } else if (node.kind == "number") {
                node.number = py::cast<std::vector<double>>(row[1]);
                require(node.number.size() == 2 && node.children.empty() &&
                        std::isfinite(node.number[0]) && std::isfinite(node.number[1]),
                        "Number nodes require two finite preprocessed features");
            } else {
                require(node.kind == "node" || node.kind == "set" || node.kind == "ordered",
                        "Unknown native tree node kind");
                node.tag = py::cast<int64_t>(row[1]);
            }
            nodes.push_back(std::move(node));
        }
        std::vector<int> color(nodes.size(), 0);
        std::unordered_map<std::string, int64_t> canonical_nodes;
        std::function<void(int64_t, int)> visit = [&](int64_t index, int depth) {
            check_root(index);
            require(depth <= 512, "Native tree nesting exceeds safe bound");
            require(color[index] != 1, "Native tree arena contains a cycle");
            if (color[index] == 2) return;
            color[index] = 1;
            auto& node = nodes[index];
            if (node.kind != "text" && node.kind != "number") {
                visit(node.tag, depth + 1);
                require(nodes[node.tag].kind == "text", "Tree tag must be a text node");
                for (auto child : node.children) visit(child, depth + 1);
            }
            // Exact structural interning is only a key for no-grad encodings.
            // Within-frame equality choices were already resolved by the checked
            // Python compiler. Binary double keys retain signed zero here.
            std::string key = node.kind + std::string(1, '\0');
            auto append_id = [&](int64_t value) {
                key.append(reinterpret_cast<const char*>(&value), sizeof(value));
            };
            if (node.kind == "text") key += node.text;
            else if (node.kind == "number")
                key.append(reinterpret_cast<const char*>(node.number.data()), node.number.size() * sizeof(double));
            else {
                append_id(nodes[node.tag].canonical);
                for (auto child : node.children) append_id(nodes[child].canonical);
            }
            auto [entry, inserted] = canonical_nodes.emplace(key, canonical_nodes.size());
            node.canonical = entry->second;
            color[index] = 2;
        };
        for (size_t i = 0; i < nodes.size(); ++i) visit(i, 0);
        for (const auto& item : raw_frames) {
            auto row = py::cast<py::dict>(item);
            Frame frame;
            frame.root = py::cast<int64_t>(row["root"]);
            require(frame.root == -1 || valid_root(frame.root), "Invalid frame semantic root");
            if (!row["map"].is_none()) {
                auto source = py::cast<py::dict>(row["map"]);
                Map map;
                map.width = py::cast<int64_t>(source["width"]);
                map.height = py::cast<int64_t>(source["height"]);
                map.own = py::cast<int64_t>(source["own"]);
                require(map.width > 0 && map.width <= 256 && map.height > 0 && map.height <= 256,
                        "Invalid native map dimensions");
                check_root(map.own);
                map.terrain = recipe_tensor(source["terrain"], at::kLong, 1, "terrain");
                map.road = recipe_tensor(source["road"], at::kLong, 1, "road");
                map.flags = recipe_tensor(source["flags"], at::kFloat, 2, "flags");
                map.attributes = recipe_tensor(source["object_attributes"], at::kFloat, 2, "object attributes");
                map.indices = recipe_tensor(source["object_indices"], at::kLong, 1, "object indices");
                const auto count = map.width * map.height;
                require(map.terrain.numel() == count && map.road.numel() == count &&
                        map.flags.sizes() == at::IntArrayRef({count, 5}) && map.attributes.size(1) == 67 &&
                        map.attributes.size(0) == map.indices.numel(), "Native map tensor shapes differ");
                require((map.terrain >= 0).all().item<bool>() && (map.terrain <= 256).all().item<bool>() &&
                        (map.road >= 0).all().item<bool>() && (map.road <= 256).all().item<bool>(),
                        "Native map embedding index is out of range");
                require((map.indices >= 0).all().item<bool>() && (map.indices < count).all().item<bool>(),
                        "Native map object index is out of range");
                frame.map = std::move(map);
            }
            frame.decision = !row["conditionals"].is_none();
            if (frame.decision) {
                require(frame.root >= 0, "A metadata frame cannot carry a decision");
                for (const auto& raw : py::cast<py::list>(row["conditionals"])) {
                    auto part = py::cast<py::dict>(raw);
                    Conditional conditional;
                    conditional.kind = py::cast<std::string>(part["kind"]);
                    conditional.selected = py::cast<int64_t>(part["selected"]);
                    require(conditional.selected >= 0, "Negative conditional selection");
                    if (conditional.kind == "categorical") {
                        conditional.roots = py::cast<std::vector<int64_t>>(part["roots"]);
                        require(!conditional.roots.empty() && conditional.selected < static_cast<int64_t>(conditional.roots.size()),
                                "Invalid categorical selection");
                        for (auto root : conditional.roots) check_root(root);
                    } else if (conditional.kind == "coordinate") {
                        conditional.root = py::cast<int64_t>(part["root"]);
                        check_root(conditional.root);
                        conditional.positions = recipe_tensor(part["positions"], at::kFloat, 2, "positions");
                        require(frame.map.has_value() && conditional.positions.size(0) == frame.map->width * frame.map->height &&
                                conditional.positions.size(1) == 2, "Invalid coordinate positions");
                        int64_t size = conditional.positions.size(0);
                        if (!part["indices"].is_none()) {
                            conditional.indices = recipe_tensor(part["indices"], at::kLong, 1, "coordinate indices");
                            require(conditional.indices.numel() > 0 && (conditional.indices >= 0).all().item<bool>() &&
                                    (conditional.indices < size).all().item<bool>(), "Invalid coordinate support indices");
                            size = conditional.indices.numel();
                        }
                        require(conditional.selected < size, "Coordinate selection exceeds support");
                    } else if (conditional.kind == "quantity") {
                        conditional.root = py::cast<int64_t>(part["root"]);
                        check_root(conditional.root);
                        conditional.numeric = recipe_tensor(part["numeric"], at::kFloat, 1, "quantity features");
                        conditional.offsets = recipe_tensor(part["offsets"], at::kFloat, 1, "quantity offsets");
                        require(conditional.numeric.numel() == 4 && conditional.offsets.numel() == 2 && conditional.selected < 2,
                                "Invalid quantity conditional");
                    } else {
                        TORCH_CHECK(false, "Unknown conditional recipe kind");
                    }
                    frame.conditionals.push_back(std::move(conditional));
                }
                require(!frame.conditionals.empty(), "A decision must contain at least one conditional");
            }
            frames.push_back(std::move(frame));
        }
    }
    bool valid_root(int64_t root) const { return root >= 0 && root < static_cast<int64_t>(nodes.size()); }
    void check_root(int64_t root) const { require(valid_root(root), "Native semantic root is out of range"); }
};

class NativeModel {
    using AutogradNode = torch::autograd::Node;
    using AutogradEdge = torch::autograd::Edge;
    struct AliasLease {
        torch::autograd::variable_list parameters;
        std::vector<std::shared_ptr<AutogradNode>> accumulators;
    };
    struct Reconnection {
        std::shared_ptr<AutogradNode> node;
        size_t edge, parameter;
    };
    struct PreparedFrame {
        Tensor vector, cells;
        Cache cache;
        AliasLease lease;
        std::vector<std::shared_ptr<AutogradNode>> nodes;
        std::vector<Reconnection> reconnections;
    };
    struct PreparedWindow {
        std::vector<PreparedFrame> frames;
        std::vector<AutogradEdge> canonical_edges;
    };
    struct BorrowedAliases {};
    std::map<std::string, Tensor> parameters;
    int64_t feature_size, hidden_size;
    std::shared_ptr<native_text::Scope> text_replay;
    native_text::Scope* prepared_text_scope = nullptr;
    torch::autograd::variable_list prepared_text_aliases;
    mutable std::vector<AliasLease> forward_alias_leases;
    NativeModel(std::map<std::string, Tensor>&& aliases, int64_t features, int64_t hidden,
                native_text::Scope* borrowed, BorrowedAliases)
        : parameters(std::move(aliases)), feature_size(features), hidden_size(hidden), prepared_text_scope(borrowed) {
        for (const auto* name : {"byte.weight", "text_gru.weight_ih_l0", "text_gru.weight_hh_l0",
                                 "text_gru.bias_ih_l0", "text_gru.bias_hh_l0"})
            prepared_text_aliases.push_back(p(name));
    }
    Tensor p(const std::string& key) const {
        auto found = parameters.find(key);
        require(found != parameters.end(), "Missing native parameter: " + key);
        return found->second;
    }
    Tensor linear(const Tensor& input, const std::string& name) const {
        return at::linear(input, p(name + ".weight"), p(name + ".bias"));
    }
    Tensor recurrent(const Tensor& input, const std::string& name) const {
        std::vector<Tensor> weights{p(name + ".weight_ih_l0"), p(name + ".weight_hh_l0"),
                                    p(name + ".bias_ih_l0"), p(name + ".bias_hh_l0")};
        auto hx = at::zeros({1, input.size(0), feature_size}, input.options());
        return std::get<1>(at::gru(input, hx, weights, true, 1, 0.0, true, false, true)).select(0, 0).select(0, 0);
    }
    Tensor encode_tree(const NativeHistory& history, int64_t root, Cache& cache, bool canonical = false) const {
        const auto key = canonical ? history.nodes[root].canonical : root;
        auto hit = cache.find(key);
        if (hit != cache.end()) return hit->second;
        const auto& node = history.nodes[root];
        Tensor result;
        if (node.kind == "text") {
            if (node.text.empty()) result = at::zeros({feature_size}, p("byte.weight").options());
            else if (prepared_text_scope && at::GradMode::is_enabled())
                result = prepared_text_scope->occurrence_prepared(node.text, feature_size, prepared_text_aliases);
            else if (text_replay && at::GradMode::is_enabled())
                result = text_replay->occurrence(node.text, feature_size);
            else {
                std::vector<int64_t> bytes;
                bytes.reserve(node.text.size());
                for (unsigned char value : node.text) bytes.push_back(static_cast<int64_t>(value) + 1);
                auto ids = at::tensor(bytes, at::TensorOptions().dtype(at::kLong));
                auto embedded = at::embedding(p("byte.weight"), ids, 0, false, false).unsqueeze(0);
                result = recurrent(embedded, "text_gru");
            }
        } else if (node.kind == "number") {
            auto numbers = at::tensor(node.number, at::TensorOptions().dtype(at::kDouble)).to(p("byte.weight").scalar_type());
            result = at::tanh(linear(numbers, "number.0"));
        } else {
            std::vector<Tensor> children;
            children.reserve(node.children.size());
            for (auto child : node.children) children.push_back(encode_tree(history, child, cache, canonical));
            Tensor pooled;
            if (children.empty()) pooled = at::zeros({feature_size}, p("byte.weight").options());
            else if (node.kind == "ordered") pooled = recurrent(at::stack(children).unsqueeze(0), "ordered");
            else pooled = at::stack(children).sum(0) / std::sqrt(static_cast<double>(children.size()));
            auto tag = encode_tree(history, node.tag, cache, canonical);
            result = at::tanh(linear(at::cat({tag, pooled}), "tree.0"));
        }
        cache.emplace(key, result);
        return result;
    }
    std::tuple<Tensor, Tensor, Tensor> encode_map(const NativeHistory& history, const Map& map, Cache& cache, bool canonical = false) const {
        const auto count = map.width * map.height;
        auto objects = at::zeros({count, 16}, p("byte.weight").options());
        if (map.indices.numel() > 0)
            objects = objects.index_add(0, map.indices, at::tanh(linear(map.attributes, "object_encoder.0")));
        auto grid = at::cat({at::embedding(p("terrain.weight"), map.terrain, 0, false, false),
                             at::embedding(p("road.weight"), map.road, 0, false, false), objects, map.flags}, -1);
        auto input = grid.reshape({map.height, map.width, 33}).permute({2, 0, 1}).unsqueeze(0);
        auto cells = at::tanh(at::conv2d(input, p("map_cnn.0.weight"), p("map_cnn.0.bias"), {1, 1}, {1, 1}));
        cells = at::tanh(at::conv2d(cells, p("map_cnn.2.weight"), p("map_cnn.2.bias"), {1, 1}, {1, 1}));
        auto global = at::tanh(linear(at::adaptive_avg_pool2d(cells, {4, 4}).flatten(1), "map_global.2")).select(0, 0);
        auto own = encode_tree(history, map.own, cache, canonical);
        return {global, own, cells.select(0, 0).permute({1, 2, 0}).reshape({-1, 16})};
    }
    std::pair<Tensor, Tensor> frame_features(const NativeHistory& history, int64_t index, Cache& cache, bool canonical = false) const {
        const auto& frame = history.frames[index];
        require(frame.root >= 0, "Metadata frames have no feature vector");
        auto zero = at::zeros({feature_size}, p("byte.weight").options());
        Tensor map = zero, own = zero, cells;
        if (frame.map) std::tie(map, own, cells) = encode_map(history, *frame.map, cache, canonical);
        auto semantics = encode_tree(history, frame.root, cache, canonical);
        return {at::cat({semantics, map, own}), cells};
    }
    Tensor transition(const Tensor& vector, const Tensor& state) const {
        return at::gru_cell(vector.unsqueeze(0), state.unsqueeze(0), p("gru.weight_ih"), p("gru.weight_hh"),
                            p("gru.bias_ih"), p("gru.bias_hh")).squeeze(0);
    }
    std::vector<Tensor> conditionals(const NativeHistory& history, const Frame& frame, const Tensor& state,
                                     const Tensor& cells, Cache& cache) const {
        std::vector<Tensor> parts;
        for (const auto& conditional : frame.conditionals) {
            Tensor logits;
            if (conditional.kind == "categorical") {
                std::vector<Tensor> candidates;
                for (auto root : conditional.roots) candidates.push_back(encode_tree(history, root, cache));
                logits = at::mv(at::stack(candidates), at::tanh(linear(state, "policy_query"))) /
                         std::sqrt(static_cast<double>(feature_size));
            } else if (conditional.kind == "coordinate") {
                const auto count = conditional.positions.size(0);
                auto target = encode_tree(history, conditional.root, cache).expand({count, -1});
                auto input = at::cat({state.expand({count, -1}), target, cells, conditional.positions}, -1);
                logits = linear(at::tanh(linear(input, "coordinate.0")), "coordinate.2").squeeze(-1);
                if (conditional.indices.defined()) logits = logits.index_select(0, conditional.indices);
            } else {
                auto features = encode_tree(history, conditional.root, cache);
                auto input = at::cat({state, features, conditional.numeric});
                logits = linear(at::tanh(linear(input, "quantity.0")), "quantity.2") + conditional.offsets;
            }
            parts.push_back(at::log_softmax(logits, 0).select(0, conditional.selected));
        }
        return parts;
    }
    void check_state(const Tensor& state) const {
        require(state.device().is_cpu() && state.scalar_type() == at::kFloat && state.sizes() == at::IntArrayRef({hidden_size}),
                "Native recurrent state must match model dimensions, CPU and float32 dtype");
    }
    static std::vector<std::string> window_texts(const NativeHistory& history, int64_t start, int64_t end,
                                                const std::vector<int64_t>& decisions) {
        std::vector<uint8_t> visited(history.nodes.size(), 0);
        std::unordered_set<std::string> seen;
        std::vector<std::string> texts;
        std::function<void(int64_t)> collect = [&](int64_t root) {
            if (visited[root]) return;
            visited[root] = 1;
            const auto& node = history.nodes[root];
            if (node.kind == "text") {
                if (!node.text.empty() && seen.insert(node.text).second) texts.push_back(node.text);
            } else if (node.kind != "number") {
                for (auto child : node.children) collect(child);
                collect(node.tag);
            }
        };
        size_t wanted = 0;
        for (int64_t index = start; index <= end; ++index) {
            const auto& frame = history.frames[index];
            if (frame.root < 0) continue;
            if (frame.map) collect(frame.map->own);
            collect(frame.root);
            if (wanted < decisions.size() && index == decisions[wanted]) {
                for (const auto& conditional : frame.conditionals) {
                    if (conditional.kind == "categorical") {
                        for (auto root : conditional.roots) collect(root);
                    } else collect(conditional.root);
                }
                ++wanted;
            }
        }
        require(wanted == decisions.size(), "Native text preparation omitted a requested decision");
        return texts;
    }
    bool parallel_frame_context() const {
        if (!text_replay || !at::GradMode::is_enabled() || at::get_num_threads() != 1 ||
            torch::autograd::get_current_graph_task_id() != -1 || at::hasCallbacks() ||
            torch::autograd::AnomalyMode::is_enabled() || !at::SavedTensorDefaultHooks::get_tls_state().stack.empty() ||
            c10::InferenceMode::is_enabled() || at::autocast::is_autocast_enabled(at::kCPU) ||
            c10::impl::TorchDispatchModeTLS::any_modes_set() || c10::impl::PythonDispatcherTLS::get_state() ||
            at::impl::PythonTorchFunctionTLS::stack_len() || at::functorch::functorchTLSAccessor()) return false;
        for (const auto& [_, parameter] : parameters)
            if (!parameter.requires_grad() || !parameter.is_leaf() || parameter.is_inference() ||
                parameter.is_nested() || parameter._is_zerotensor() || at::isTensorSubclassLike(parameter) ||
                parameter.layout() != at::kStrided || parameter._fw_grad(0).defined() ||
                native_text::parameter_hooks(parameter)) return false;
        return true;
    }
    static bool supported_frame_node(const AutogradNode& node) {
        using namespace torch::autograd::generated;
        const auto& type = typeid(node);
        return type == typeid(native_text::Occurrence) || type == typeid(AddBackward0) ||
            type == typeid(AddmmBackward0) || type == typeid(MmBackward0) || type == typeid(MulBackward0) ||
            type == typeid(SubBackward0) || type == typeid(DivBackward0) || type == typeid(DivBackward1) ||
            type == typeid(SigmoidBackward0) || type == typeid(TanhBackward0) || type == typeid(EmbeddingBackward0) ||
            type == typeid(ViewBackward0) || type == typeid(UnsafeViewBackward0) || type == typeid(ReshapeAliasBackward0) ||
            type == typeid(SelectBackward0) || type == typeid(TBackward0) || type == typeid(TransposeBackward0) ||
            type == typeid(UnsqueezeBackward0) || type == typeid(UnbindBackward0) || type == typeid(UnsafeSplitBackward0) ||
            type == typeid(StackBackward0) || type == typeid(SumBackward1) || type == typeid(CatBackward0) ||
            type == typeid(PermuteBackward0) || type == typeid(CloneBackward0) || type == typeid(IndexAddBackward0) ||
            type == typeid(SqueezeBackward0) || type == typeid(SqueezeBackward1) || type == typeid(SqueezeBackward2) ||
            type == typeid(SqueezeBackward3) || type == typeid(SqueezeBackward4) || type == typeid(SqueezeBackward5) ||
            type == typeid(ExpandBackward0) || type == typeid(AsStridedBackward0) ||
            type == typeid(ConvolutionBackward0) || type == typeid(AdaptiveAvgPool2DBackward0);
    }
    PreparedFrame prepare_frame(const NativeHistory& history, int64_t index) const {
        PreparedFrame result;
        std::map<std::string, Tensor> aliases;
        std::unordered_map<AutogradNode*, size_t> terminals;
        for (const auto& [name, parameter] : parameters) {
            auto alias = parameter.detach().set_requires_grad(true);
            const auto edge = torch::autograd::impl::gradient_edge(alias);
            require(edge.function && edge.input_nr == 0 &&
                    typeid(*edge.function) == typeid(torch::autograd::AccumulateGrad),
                    "Native frame alias has an unexpected accumulator");
            terminals.emplace(edge.function.get(), result.lease.parameters.size());
            result.lease.parameters.push_back(alias);
            result.lease.accumulators.push_back(edge.function);
            aliases.emplace(name, std::move(alias));
        }
        NativeModel view(std::move(aliases), feature_size, hidden_size, text_replay.get(), BorrowedAliases{});
        std::tie(result.vector, result.cells) = view.frame_features(history, index, result.cache);
        std::vector<std::shared_ptr<AutogradNode>> pending;
        auto add_root = [&](const Tensor& tensor) {
            if (tensor.defined() && tensor.grad_fn()) pending.push_back(tensor.grad_fn());
        };
        add_root(result.vector); add_root(result.cells);
        for (const auto& [_, tensor] : result.cache) add_root(tensor);
        std::unordered_set<AutogradNode*> seen;
        const auto thread = at::RecordFunction::currentThreadId();
        while (!pending.empty()) {
            auto node = std::move(pending.back()); pending.pop_back();
            if (!seen.insert(node.get()).second) continue;
            require(!terminals.count(node.get()) && supported_frame_node(*node) && node->thread_id() == thread &&
                    node->pre_hooks().empty() && node->post_hooks().empty() && node->tensor_pre_hooks().empty() &&
                    node->retains_grad_hooks().empty(),
                    "Native frame graph has unsupported nodes, hooks or ownership: " + node->name());
            for (size_t edge_index = 0; edge_index < node->next_edges().size(); ++edge_index) {
                const auto& edge = node->next_edge(edge_index);
                if (!edge.is_valid()) continue;
                const auto terminal = terminals.find(edge.function.get());
                if (terminal != terminals.end()) {
                    require(edge.input_nr == 0, "Native frame alias edge has an invalid input");
                    result.reconnections.push_back({node, edge_index, terminal->second});
                } else pending.push_back(edge.function);
            }
            result.nodes.push_back(std::move(node));
        }
        std::sort(result.nodes.begin(), result.nodes.end(), [](const auto& a, const auto& b) {
            return a->sequence_nr() < b->sequence_nr();
        });
        for (size_t i = 1; i < result.nodes.size(); ++i)
            require(result.nodes[i-1]->sequence_nr() != result.nodes[i]->sequence_nr(),
                    "Native frame graph has ambiguous backward priorities");
        return result;
    }
    PreparedWindow prepare_frames(const NativeHistory& history, int64_t start, int64_t end,
                                   const std::vector<int64_t>& jobs, int workers) const {
        std::vector<native_text::ParameterState> states;
        for (const auto& [_, parameter] : parameters) states.emplace_back(parameter);
        PreparedWindow result;
        result.frames.resize(end - start + 1);
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        std::mutex error_mutex;
        const at::ThreadLocalState thread_state;
        auto run = [&] {
            try {
                at::get_num_threads(); at::set_num_threads(1);
                require(at::get_num_threads() == 1, "Native frame worker initialization failed");
                at::ThreadLocalStateGuard state_guard(thread_state);
                while (!failed.load()) {
                    const auto job = next.fetch_add(1);
                    if (job >= jobs.size()) break;
                    result.frames[jobs[job] - start] = prepare_frame(history, jobs[job]);
                }
            } catch (...) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        };
        std::vector<std::thread> threads;
        try {
            const int count = std::min<int>(workers, jobs.size());
            for (int i = 1; i < count; ++i) threads.emplace_back(run);
            run();
        } catch (...) {
            failed.store(true);
            for (auto& thread : threads) if (thread.joinable()) thread.join();
            throw;
        }
        for (auto& thread : threads) thread.join();
        if (error) std::rethrow_exception(error);
        size_t parameter_index = 0;
        for (const auto& [_, parameter] : parameters) {
            require(states[parameter_index++].matches(parameter) && !native_text::parameter_hooks(parameter),
                    "Native parameters changed during parallel frame preparation");
            auto edge = torch::autograd::impl::gradient_edge(parameter);
            require(edge.function && edge.input_nr == 0 &&
                    typeid(*edge.function) == typeid(torch::autograd::AccumulateGrad) &&
                    edge.function->topological_nr() == 0, "Native canonical parameter has an invalid accumulator");
            result.canonical_edges.push_back(std::move(edge));
        }
        // Validate every substitution before changing any prepared graph. Both
        // terminals have depth zero, so replacing an edge cannot change parent
        // topology. Public edge assignment avoids the setter's parented-node guard.
        for (const auto& frame : result.frames) for (const auto& link : frame.reconnections) {
            const auto& alias = frame.lease.parameters[link.parameter];
            const auto& old = frame.lease.accumulators[link.parameter];
            const auto& target = result.canonical_edges[link.parameter];
            const auto canonical = std::static_pointer_cast<torch::autograd::AccumulateGrad>(target.function)->variable;
            require(link.node->next_edge(link.edge).function == old && link.node->next_edge(link.edge).input_nr == 0 &&
                    old->topological_nr() == 0 && old->num_inputs() == 1 && target.function->num_inputs() == 1 &&
                    !alias.is_same(canonical) && alias.const_data_ptr() == canonical.const_data_ptr() &&
                    alias.storage().unsafeGetStorageImpl() == canonical.storage().unsafeGetStorageImpl() &&
                    alias.scalar_type() == canonical.scalar_type() && alias.sizes() == canonical.sizes() &&
                    alias.strides() == canonical.strides() && old->input_metadata(0).is_same_shape(canonical) &&
                    target.function->input_metadata(0).is_same_shape(alias),
                    "Native frame terminal substitution is not equivalent");
        }
        return result;
    }
public:
    ~NativeModel() noexcept {
        // Destruction joins before releasing weak-accumulator leases.
        if (text_replay) text_replay->close(false);
        forward_alias_leases.clear();
    }
    void begin_text_replay(int workers) {
        require(!text_replay, "A native text replay scope is already open");
        require(forward_alias_leases.empty(), "Native frame aliases outlived their replay scope");
        std::vector<Tensor> values{p("byte.weight"), p("text_gru.weight_ih_l0"), p("text_gru.weight_hh_l0"),
                                  p("text_gru.bias_ih_l0"), p("text_gru.bias_hh_l0")};
        text_replay = std::make_shared<native_text::Scope>(std::move(values), workers);
    }
    void backward_with_text_replay(const Tensor& loss) {
        require(static_cast<bool>(text_replay), "Native text backward requires an open replay scope");
        text_replay->backward(loss);
    }
    void end_text_replay() {
        if (!text_replay) return;
        auto scope = std::move(text_replay);
        try {
            scope->close();
        } catch (...) {
            forward_alias_leases.clear();
            throw;
        }
        forward_alias_leases.clear();
    }
    void close() {
        if (!text_replay) { forward_alias_leases.clear(); return; }
        auto scope = std::move(text_replay);
        // Cleanup joins every worker even after failure. The strict scope-end
        // operation reports unfinished work; cleanup preserves the primary error.
        scope->close(false);
        forward_alias_leases.clear();
    }
    NativeModel(const std::map<std::string, Tensor>& named, int64_t features, int64_t hidden)
        : parameters(named), feature_size(features), hidden_size(hidden) {
        require(feature_size > 0 && hidden_size > 0, "Native dimensions must be positive");
        std::map<std::string, std::vector<int64_t>> expected;
        auto linear_shape = [&](const std::string& name, int64_t in, int64_t out) {
            expected[name + ".weight"] = {out, in}; expected[name + ".bias"] = {out};
        };
        auto gru_shape = [&](const std::string& name, int64_t in) {
            expected[name + ".weight_ih_l0"] = {3 * features, in};
            expected[name + ".weight_hh_l0"] = {3 * features, features};
            expected[name + ".bias_ih_l0"] = {3 * features}; expected[name + ".bias_hh_l0"] = {3 * features};
        };
        expected["byte.weight"] = {257, 8}; expected["terrain.weight"] = {257, 8}; expected["road.weight"] = {257, 4};
        gru_shape("text_gru", 8); gru_shape("ordered", features);
        linear_shape("number.0", 2, features); linear_shape("tree.0", 2 * features, features);
        linear_shape("object_encoder.0", 67, 16); linear_shape("map_global.2", 256, features);
        expected["map_cnn.0.weight"] = {16, 33, 3, 3}; expected["map_cnn.0.bias"] = {16};
        expected["map_cnn.2.weight"] = {16, 16, 3, 3}; expected["map_cnn.2.bias"] = {16};
        expected["gru.weight_ih"] = {3 * hidden, 3 * features}; expected["gru.weight_hh"] = {3 * hidden, hidden};
        expected["gru.bias_ih"] = {3 * hidden}; expected["gru.bias_hh"] = {3 * hidden};
        linear_shape("policy_query", hidden, features); linear_shape("value", hidden, 1);
        linear_shape("coordinate.0", hidden + features + 18, features); linear_shape("coordinate.2", features, 1);
        linear_shape("quantity.0", hidden + features + 4, features); linear_shape("quantity.2", features, 2);
        require(parameters.size() == expected.size(), "Native parameter inventory differs from the reference architecture");
        std::unordered_set<c10::TensorImpl*> identities;
        for (const auto& [name, shape] : expected) {
            const auto tensor = p(name);
            require(identities.insert(tensor.unsafeGetTensorImpl()).second,
                    "Native model requires distinct parameter identities: " + name);
            require(tensor.device().is_cpu() && tensor.scalar_type() == at::kFloat && tensor.sizes() == at::IntArrayRef(shape),
                    "Native parameter shape, dtype or device differs: " + name);
            require(at::isfinite(tensor).all().item<bool>(), "Nonfinite native parameter: " + name);
        }
    }
    Tensor tree(const NativeHistory& history, int64_t root) const {
        history.check_root(root); Cache cache; return encode_tree(history, root, cache);
    }
    std::pair<Tensor, Tensor> encode_frame(const NativeHistory& history, int64_t index) const {
        require(index >= 0 && index < static_cast<int64_t>(history.frames.size()), "Frame index out of range");
        Cache cache; return frame_features(history, index, cache);
    }
    std::map<int64_t, Tensor> prefix(const NativeHistory& history, const std::vector<int64_t>& stops, int workers) const {
        require(!stops.empty() && workers >= 1 && workers <= 256, "Prefix needs boundaries and 1 to 256 workers");
        int64_t end = 0;
        for (auto stop : stops) {
            require(stop >= 0 && stop <= static_cast<int64_t>(history.frames.size()), "Prefix boundary out of range");
            end = std::max(end, stop);
        }
        require(at::get_num_threads() == 1,
                "Native prefixes require one ATen intra-op thread to preserve the accepted kernel contract");
        at::NoGradGuard no_grad;
        std::vector<Tensor> vectors(end);
        std::vector<int64_t> jobs;
        for (int64_t i = 0; i < end; ++i) if (history.frames[i].root >= 0) jobs.push_back(i);
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        std::mutex error_mutex;
        const at::ThreadLocalState thread_state;
        auto run = [&] {
            try {
                // Intra-op settings are not part of ATen ThreadLocalState.
                // Initialize each fresh thread before setting its validated count.
                at::get_num_threads();
                at::set_num_threads(1);
                require(at::get_num_threads() == 1, "Native prefix thread count changed");
                at::ThreadLocalStateGuard state_guard(thread_state);
                at::NoGradGuard guard;
                Cache cache;  // Current weights, this prefix call, this worker only.
                while (!failed.load()) {
                    const auto job = next.fetch_add(1);
                    if (job >= jobs.size()) break;
                    const auto index = jobs[job];
                    if (cache.size() >= 65536) cache.clear();
                    vectors[index] = frame_features(history, index, cache, true).first;
                }
            } catch (...) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        };
        std::vector<std::thread> threads;
        try {
            const int count = std::min<int>(workers, std::max<size_t>(1, jobs.size()));
            for (int i = 1; i < count; ++i) threads.emplace_back(run);
            run();
        } catch (...) {
            failed.store(true);
            for (auto& thread : threads) if (thread.joinable()) thread.join();
            throw;
        }
        for (auto& thread : threads) thread.join();
        if (error) std::rethrow_exception(error);
        std::map<int64_t, Tensor> result;
        auto state = at::zeros({hidden_size}, p("byte.weight").options());
        for (auto stop : stops) if (stop == 0) result[0] = state.detach();
        for (int64_t i = 0; i < end; ++i) {
            if (vectors[i].defined()) state = transition(vectors[i], state);
            if (std::find(stops.begin(), stops.end(), i + 1) != stops.end()) result[i + 1] = state.detach();
        }
        return result;
    }
    void prepare_window_text(const NativeHistory& history, int64_t start, int64_t end,
                             const std::vector<int64_t>& decisions, int workers) {
        require(static_cast<bool>(text_replay), "Native text preparation requires an open replay scope");
        require(workers >= 1 && workers <= 256, "Native text preparation needs 1 to 256 workers");
        require(start >= 0 && start <= end && end < static_cast<int64_t>(history.frames.size()) && !decisions.empty(),
                "Invalid native loss-window boundary");
        int64_t previous = start - 1;
        for (auto index : decisions) {
            require(index > previous && index >= start && index <= end && history.frames[index].decision,
                    "Native decisions must be ordered and have recorded conditional recipes");
            previous = index;
        }
        // Prewarming creates only the alias-owned inner leaf graphs.
        text_replay->prewarm(window_texts(history, start, end, decisions), feature_size, workers);
    }
    std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> evaluate_window_details(
            const NativeHistory& history, int64_t start, int64_t end,
            const std::vector<int64_t>& decisions, const Tensor& initial, int workers = 1) const {
        require(workers >= 1 && workers <= 256, "Native loss window needs 1 to 256 workers");
        require(start >= 0 && start <= end && end < static_cast<int64_t>(history.frames.size()) && !decisions.empty(),
                "Invalid native loss-window boundary");
        check_state(initial);
        int64_t previous = start - 1;
        for (auto index : decisions) {
            require(index > previous && index >= start && index <= end && history.frames[index].decision,
                    "Native decisions must be ordered and have recorded conditional recipes");
            previous = index;
        }
        std::optional<PreparedWindow> prepared;
        if (workers > 1 && !initial._fw_grad(0).defined() && parallel_frame_context()) {
            std::vector<int64_t> jobs;
            for (int64_t index = start; index <= end; ++index)
                if (history.frames[index].root >= 0) jobs.push_back(index);
            if (jobs.size() > 1) {
                text_replay->require_prepared(window_texts(history, start, end, decisions), feature_size);
                prepared = prepare_frames(history, start, end, jobs, workers);
                forward_alias_leases.reserve(forward_alias_leases.size() + jobs.size());
            }
        }
        std::vector<Tensor> log_probs, values, conditional_values;
        auto state = initial.detach();
        size_t wanted = 0;
        for (int64_t index = start; index <= end; ++index) {
            const auto& frame = history.frames[index];
            if (frame.root < 0) continue;
            Cache cache;
            Tensor vector, cells;
            if (prepared) {
                auto& features = prepared->frames[index - start];
                for (const auto& link : features.reconnections)
                    link.node->next_edges()[link.edge] = prepared->canonical_edges[link.parameter];
                for (const auto& node : features.nodes)
                    node->set_sequence_nr(at::sequence_number::get_and_increment());
                // SavedVariable keeps weak alias accumulators. Retain their
                // owners through backward even after redirecting outer edges.
                forward_alias_leases.push_back(std::move(features.lease));
                vector = std::move(features.vector); cells = std::move(features.cells);
                cache = std::move(features.cache);
                features.nodes.clear(); features.reconnections.clear();
            } else std::tie(vector, cells) = frame_features(history, index, cache);
            state = transition(vector, state);
            // The reference constructs the value head for every decision, but unused
            // head graphs do not contribute a gradient. Preserve even those calls.
            Tensor value;
            if (frame.decision) value = linear(state, "value").squeeze(-1);
            if (wanted < decisions.size() && index == decisions[wanted]) {
                auto parts = conditionals(history, frame, state, cells, cache);
                auto combined = at::stack(parts);
                conditional_values.push_back(combined);
                log_probs.push_back(combined.sum());
                values.push_back(value);
                ++wanted;
            }
        }
        require(wanted == decisions.size(), "Native loss window omitted a requested decision");
        return {at::stack(log_probs), at::stack(values), state, conditional_values};
    }
    std::tuple<Tensor, Tensor, Tensor> evaluate_window(const NativeHistory& history, int64_t start, int64_t end,
            const std::vector<int64_t>& decisions, const Tensor& initial, int workers = 1) const {
        auto [log_probs, values, state, details] = evaluate_window_details(history, start, end, decisions, initial, workers);
        return {log_probs, values, state};
    }
};

std::vector<Tensor> clipped_losses(const Tensor& log_probs, const Tensor& values,
        const Tensor& old_log_probs_input, const Tensor& old_values_input,
        const Tensor& advantages_input, const Tensor& returns_input,
        double clip_ratio, std::optional<double> value_clip, double value_coefficient) {
    require(log_probs.dim() == 1 && log_probs.numel() > 0, "PPO loss requires nonempty vectors");
    require(std::isfinite(clip_ratio) && clip_ratio > 0 && clip_ratio < 1 &&
            std::isfinite(value_coefficient) && value_coefficient >= 0 &&
            (!value_clip || (std::isfinite(*value_clip) && *value_clip > 0)), "Invalid native PPO coefficients");
    for (const auto& tensor : {log_probs, values, old_log_probs_input, old_values_input, advantages_input, returns_input}) {
        require(tensor.sizes() == log_probs.sizes() && tensor.device().is_cpu() && tensor.scalar_type() == log_probs.scalar_type(),
                "Native PPO inputs must be aligned CPU vectors");
        require(at::isfinite(tensor).all().item<bool>(), "Nonfinite native PPO input");
    }
    const auto old_log_probs = old_log_probs_input.detach(), old_values = old_values_input.detach();
    const auto advantages = advantages_input.detach(), returns = returns_input.detach();
    auto log_ratio = log_probs - old_log_probs;
    auto ratio = log_ratio.exp();
    require(at::isfinite(ratio).all().item<bool>(), "Nonfinite native PPO importance ratio");
    auto policy = -at::minimum(ratio * advantages, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages).mean();
    auto squared = (values - returns).square();
    if (value_clip) {
        auto clipped = old_values + (values - old_values).clamp(-*value_clip, *value_clip);
        squared = at::maximum(squared, (clipped - returns).square());
    }
    auto value = 0.5 * squared.mean();
    auto total = policy + value_coefficient * value;
    return {total, policy, value, ((ratio - 1) - log_ratio).mean(), ((ratio - 1).abs() > clip_ratio).to(at::kFloat).mean()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("contract") = "heroes-native-learner-v1";
    // Diagnostic surface for exact Engine-versus-tape comparison on private graphs.
    using Tape = native_text::BuiltinReplayTape;
    py::class_<Tape, std::shared_ptr<Tape>>(module, "_ReplayTape")
        .def_static("compile", [](const Tensor& output, const torch::autograd::variable_list& aliases) {
            std::string reason;
            auto tape = Tape::compile(output, aliases, {}, &reason);
            return std::make_pair(std::move(tape), std::move(reason));
        }, py::call_guard<py::gil_scoped_release>())
        .def("run", [](const Tape& tape, const Tensor& gradient) {
            torch::autograd::variable_list captured;
            const bool ran = tape.run_if_supported(gradient, &captured);
            return std::make_pair(ran, std::move(captured));
        }, py::call_guard<py::gil_scoped_release>())
        .def("supported_now", [](const Tape& tape) {
            std::string reason;
            const bool supported = tape.supported_now(&reason);
            return std::make_pair(supported, std::move(reason));
        }, py::call_guard<py::gil_scoped_release>())
        .def("node_count", &Tape::node_count);
    module.def("_text_replay_probe", [](const torch::autograd::variable_list& parameters,
                                       const std::string& text, int64_t features,
                                       const torch::autograd::variable_list& gradients) {
        require(parameters.size() == 5 && !text.empty() && features > 0 && !gradients.empty(),
                "A text replay probe requires five parameters, text, features and gradients");
        require(at::get_num_threads() == 1, "A text replay probe requires one ATen intra-op thread");
        for (const auto& parameter : parameters)
            require(parameter.defined() && parameter.device().is_cpu() && parameter.scalar_type() == at::kFloat &&
                    parameter.layout() == at::kStrided && parameter.is_leaf() && parameter.requires_grad() &&
                    !native_text::parameter_hooks(parameter), "A text replay probe requires ordinary trainable parameters");
        auto leaf = native_text::Leaf::make(parameters, text, features);
        struct Cleanup {
            std::shared_ptr<native_text::Leaf> leaf;
            ~Cleanup() { leaf->release(); }
        } cleanup{leaf};
        std::vector<torch::autograd::variable_list> reference, actual;
        for (const auto& gradient : gradients) {
            reference.push_back(leaf->replay(gradient, false));
            actual.push_back(leaf->replay(gradient));
        }
        return std::make_tuple(std::move(reference), std::move(actual),
                               static_cast<bool>(leaf->tape), leaf->tape_fallback_reason);
    }, py::call_guard<py::gil_scoped_release>());
    py::class_<NativeHistory>(module, "NativeHistory")
        .def(py::init<const py::list&, const py::list&>())
        .def_property_readonly("frame_count", [](const NativeHistory& value) { return value.frames.size(); });
    py::class_<NativeModel>(module, "NativeModel")
        .def(py::init<const std::map<std::string, Tensor>&, int64_t, int64_t>())
        .def("begin_text_replay", &NativeModel::begin_text_replay, py::call_guard<py::gil_scoped_release>())
        .def("backward_with_text_replay", &NativeModel::backward_with_text_replay, py::call_guard<py::gil_scoped_release>())
        .def("end_text_replay", &NativeModel::end_text_replay, py::call_guard<py::gil_scoped_release>())
        .def("close", &NativeModel::close, py::call_guard<py::gil_scoped_release>())
        .def("tree", &NativeModel::tree, py::call_guard<py::gil_scoped_release>())
        .def("encode_frame", &NativeModel::encode_frame, py::call_guard<py::gil_scoped_release>())
        .def("prefix", &NativeModel::prefix, py::arg("history"), py::arg("stops"), py::arg("workers") = 1,
             py::call_guard<py::gil_scoped_release>())
        .def("prepare_window_text", &NativeModel::prepare_window_text, py::call_guard<py::gil_scoped_release>())
        .def("evaluate_window", &NativeModel::evaluate_window, py::arg("history"), py::arg("start"), py::arg("end"),
             py::arg("decisions"), py::arg("initial"), py::arg("workers") = 1, py::call_guard<py::gil_scoped_release>())
        .def("evaluate_window_details", &NativeModel::evaluate_window_details, py::arg("history"), py::arg("start"), py::arg("end"),
             py::arg("decisions"), py::arg("initial"), py::arg("workers") = 1, py::call_guard<py::gil_scoped_release>());
    module.def("clipped_losses", &clipped_losses, py::call_guard<py::gil_scoped_release>());
}
