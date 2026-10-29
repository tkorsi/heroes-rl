// Exact text-leaf graph reuse and native parallel backward replay.
// Each occurrence returns every parameter-edge contribution separately. Deferred
// replays fold those contributions in outer-engine order, never in worker order.
#pragma once

#include "replay_tape.h"
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/graph_task.h>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <typeindex>
#include <unordered_set>

namespace native_text {
using torch::autograd::Node;
using torch::autograd::variable_list;
using ReplayClock = std::chrono::steady_clock;

struct WallTimer {
    int64_t& target;
    ReplayClock::time_point started = ReplayClock::now();
    explicit WallTimer(int64_t& value) : target(value) {}
    ~WallTimer() {
        target += std::chrono::duration_cast<std::chrono::nanoseconds>(ReplayClock::now() - started).count();
    }
};

struct ParameterState {
    const void* data;
    c10::TensorImpl* identity;
    uint32_t version;
    at::ScalarType dtype;
    std::vector<int64_t> sizes, strides;
    explicit ParameterState(const Tensor& value)
        : data(value.const_data_ptr()), identity(value.unsafeGetTensorImpl()),
          version(value.unsafeGetTensorImpl()->version_counter().current_version()),
          dtype(value.scalar_type()), sizes(value.sizes().vec()), strides(value.strides().vec()) {}
    bool matches(const Tensor& value) const {
        return value.requires_grad() && value.unsafeGetTensorImpl() == identity &&
            value.const_data_ptr() == data && value.scalar_type() == dtype &&
            value.sizes() == at::IntArrayRef(sizes) && value.strides() == at::IntArrayRef(strides) &&
            value.unsafeGetTensorImpl()->version_counter().current_version() == version;
    }
};

inline bool parameter_hooks(const Tensor& value) {
    using namespace torch::autograd;
    if (!impl::hooks(value).empty() || impl::post_acc_grad_hooks(value)) return true;
    const auto node = impl::gradient_edge(value).function;
    return node && (!node->pre_hooks().empty() || !node->post_hooks().empty());
}

inline bool parameter_hooks_readonly(const Tensor& value) {
    using namespace torch::autograd;
    if (!impl::hooks(value).empty() || impl::post_acc_grad_hooks(value)) return true;
    // Worker validation must not create a canonical AccumulateGrad node.
    const auto node = impl::try_get_grad_accumulator(value);
    return node && (!node->pre_hooks().empty() || !node->post_hooks().empty());
}

struct Leaf;
struct Capture {
    const Leaf* leaf;
    variable_list* contributions;
    std::vector<size_t>* cursors;
    const std::vector<size_t>* offsets;
};
inline thread_local Capture* current_capture = nullptr;

struct ContributionHook final : torch::autograd::FunctionPostHook {
    std::weak_ptr<Leaf> owner;
    std::vector<std::pair<size_t, size_t>> targets;
    ContributionHook(std::weak_ptr<Leaf> leaf, std::vector<std::pair<size_t, size_t>> slots)
        : owner(std::move(leaf)), targets(std::move(slots)) {}
    variable_list operator()(const variable_list& inputs, const variable_list&) override {
        const auto leaf = owner.lock();
        TORCH_CHECK(leaf && current_capture && current_capture->leaf == leaf.get(),
                    "A native text leaf graph ran outside its own replay");
        auto result = inputs;
        for (const auto [edge, position] : targets) {
            TORCH_CHECK(edge < result.size(), "Native text gradient edge is missing");
            TORCH_CHECK(position < current_capture->cursors->size(), "Native text gradient parameter is missing");
            auto& cursor = (*current_capture->cursors)[position];
            TORCH_CHECK(cursor < (*current_capture->offsets)[position + 1],
                        "Native text replay contribution count increased");
            (*current_capture->contributions)[cursor++] = result[edge];
            result[edge] = Tensor();  // The outer occurrence, not this engine, adds each term.
        }
        return result;
    }
};

struct Leaf : std::enable_shared_from_this<Leaf> {
    variable_list parameters, aliases, arguments;
    std::vector<ParameterState> states;
    std::vector<size_t> counts, offsets;
    Tensor output, value;
    std::shared_ptr<BuiltinReplayTape> tape;
    std::string tape_fallback_reason;
    mutable std::atomic<int64_t> tape_replays{0}, engine_replays{0};

    static std::shared_ptr<Leaf> make(const variable_list& parameters, const std::string& text, int64_t features) {
        auto leaf = std::make_shared<Leaf>();
        leaf->parameters = parameters;
        leaf->counts.resize(parameters.size(), 0);
        for (const auto& parameter : parameters) {
            leaf->states.emplace_back(parameter);
            leaf->aliases.push_back(parameter.detach().set_requires_grad(true));
        }
        std::vector<int64_t> bytes;
        bytes.reserve(text.size());
        for (unsigned char byte : text) bytes.push_back(static_cast<int64_t>(byte) + 1);
        {
            at::AutoGradMode enable(true);
            auto index = at::tensor(bytes, at::TensorOptions().dtype(at::kLong));
            auto embedding = at::embedding(leaf->aliases[0], index, 0, false, false).unsqueeze(0);
            auto hx = at::zeros({1, 1, features}, embedding.options());
            variable_list weights(leaf->aliases.begin() + 1, leaf->aliases.end());
            leaf->output = std::get<1>(at::gru(embedding, hx, weights, true, 1, 0.0, true, false, true))
                               .select(0, 0).select(0, 0);
        }
        leaf->value = leaf->output.detach();
        std::unordered_map<c10::TensorImpl*, size_t> slots;
        for (size_t i = 0; i < leaf->aliases.size(); ++i) slots[leaf->aliases[i].unsafeGetTensorImpl()] = i;
        std::unordered_set<Node*> seen;
        std::unordered_set<const torch::autograd::FunctionPostHook*> allowed_post_hooks;
        std::vector<std::shared_ptr<Node>> pending{leaf->output.grad_fn()};
        while (!pending.empty()) {
            auto node = pending.back();
            pending.pop_back();
            if (!node || !seen.insert(node.get()).second) continue;
            std::vector<std::pair<size_t, size_t>> targets;
            const auto& edges = node->next_edges();
            for (size_t i = 0; i < edges.size(); ++i) {
                const auto child = edges[i].function;
                const auto accumulator = std::dynamic_pointer_cast<torch::autograd::AccumulateGrad>(child);
                auto slot = accumulator ? slots.find(accumulator->variable.unsafeGetTensorImpl()) : slots.end();
                if (slot != slots.end()) {
                    targets.emplace_back(i, slot->second);
                    ++leaf->counts[slot->second];
                } else if (child) pending.push_back(child);
            }
            if (!targets.empty()) {
                auto hook = std::make_unique<ContributionHook>(leaf, std::move(targets));
                allowed_post_hooks.insert(hook.get());
                node->add_post_hook(std::move(hook));
            }
        }
        leaf->offsets.push_back(0);
        for (size_t i = 0; i < parameters.size(); ++i) {
            for (size_t j = 0; j < leaf->counts[i]; ++j) leaf->arguments.push_back(parameters[i]);
            leaf->offsets.push_back(leaf->arguments.size());
        }
        leaf->tape = BuiltinReplayTape::compile(leaf->output, leaf->aliases, allowed_post_hooks,
                                               &leaf->tape_fallback_reason);
        return leaf;
    }

    void check() const {
        TORCH_CHECK(value.defined(), "A released native text leaf cannot be replayed");
        for (size_t i = 0; i < parameters.size(); ++i)
            TORCH_CHECK(states[i].matches(parameters[i]), "Native text parameters changed inside a replay scope");
    }

    variable_list replay(const Tensor& gradient, bool allow_tape = true) const {
        check();
        TORCH_CHECK(at::get_num_threads() == 1, "Native text replay requires one ATen intra-op thread");
        variable_list contributions(arguments.size());
        std::vector<size_t> cursors(offsets.begin(), offsets.end() - 1);
        Capture capture{this, &contributions, &cursors, &offsets};
        struct CaptureGuard {
            Capture* previous;
            explicit CaptureGuard(Capture* capture) : previous(current_capture) { current_capture = capture; }
            ~CaptureGuard() { current_capture = previous; }
        } guard(&capture);
        // False is a pre-execution eligibility decision. Execution failures must
        // propagate without retrying a graph whose hooks may already have run.
        const bool used_tape = allow_tape && tape && tape->run_if_supported(gradient);
        if (!used_tape) torch::autograd::grad({output}, aliases, {gradient}, true, false, true);
        for (size_t i = 0; i < parameters.size(); ++i)
            TORCH_CHECK(cursors[i] == offsets[i + 1], "Native text replay contribution count changed");
        if (used_tape) tape_replays.fetch_add(1, std::memory_order_relaxed);
        else engine_replays.fetch_add(1, std::memory_order_relaxed);
        return contributions;
    }

    void release() { tape.reset(); output = value = Tensor(); aliases.clear(); }
};

struct ReplayJob {
    std::shared_ptr<Leaf> leaf;
    Tensor gradient;
    std::atomic<int> state{0};  // queued, claimed, completed
    std::promise<variable_list> promise;
    std::future<variable_list> future;
    ReplayJob(std::shared_ptr<Leaf> item, Tensor grad)
        : leaf(std::move(item)), gradient(std::move(grad)), future(promise.get_future()) {}
};

class ReplayPool {
    std::mutex mutex;
    std::condition_variable available, ready;
    std::deque<std::shared_ptr<ReplayJob>> queue;
    std::vector<std::thread> threads;
    bool stopping = false;
    int initialized = 0;
    std::exception_ptr startup_error;
    at::ThreadLocalState thread_state;
    std::atomic<int64_t> active{0}, peak{0}, completed{0}, helped{0};

    void worker() {
        std::optional<at::ThreadLocalStateGuard> state_guard;
        try {
            // ATen intra-op settings are not included in ThreadLocalState.
            at::get_num_threads();
            at::set_num_threads(1);
            TORCH_CHECK(at::get_num_threads() == 1, "Native replay worker initialization failed");
            state_guard.emplace(thread_state);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!startup_error) startup_error = std::current_exception();
            ++initialized;
            ready.notify_all();
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++initialized;
            ready.notify_all();
        }
        for (;;) {
            std::shared_ptr<ReplayJob> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                available.wait(lock, [&] { return stopping || !queue.empty(); });
                if (stopping) return;
                job = queue.front();
                queue.pop_front();
            }
            run(job, false);
        }
    }

public:
    explicit ReplayPool(int workers) {
        try {
            for (int i = 0; i < workers; ++i) threads.emplace_back([this] { worker(); });
            std::unique_lock<std::mutex> lock(mutex);
            ready.wait(lock, [&] { return initialized == workers; });
            if (startup_error) std::rethrow_exception(startup_error);
        } catch (...) {
            close();
            throw;
        }
    }
    ~ReplayPool() { close(); }
    void submit(const std::shared_ptr<ReplayJob>& job) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            TORCH_CHECK(!stopping, "Native text replay pool is closed");
            queue.push_back(job);
        }
        available.notify_one();
    }
    bool run(const std::shared_ptr<ReplayJob>& job, bool on_main) {
        int expected = 0;
        if (!job->state.compare_exchange_strong(expected, 1)) return false;
        const auto running = active.fetch_add(1) + 1;
        auto maximum = peak.load();
        while (running > maximum && !peak.compare_exchange_weak(maximum, running)) {}
        try {
            job->promise.set_value(job->leaf->replay(job->gradient));
        } catch (...) {
            job->promise.set_exception(std::current_exception());
        }
        job->gradient = Tensor();
        active.fetch_sub(1);
        completed.fetch_add(1);
        if (on_main) helped.fetch_add(1);
        job->state.store(2);
        return true;
    }
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            queue.clear();
        }
        available.notify_all();
        for (auto& thread : threads) if (thread.joinable()) thread.join();
        threads.clear();
    }
    std::map<std::string, int64_t> statistics() const {
        return {{"peak_parallel_replays", peak.load()}, {"completed_replays", completed.load()},
                {"main_thread_replays", helped.load()}};
    }
};

struct Scope;
struct Occurrence final : Node {
    std::shared_ptr<Leaf> leaf;
    std::shared_ptr<Scope> scope;
    size_t compact_edges = 0;
    Occurrence(std::shared_ptr<Leaf> item, std::shared_ptr<Scope> context)
        : Node(torch::autograd::collect_next_edges(item->arguments)), leaf(std::move(item)), scope(std::move(context)) {}
    Occurrence(std::shared_ptr<Leaf> item, std::shared_ptr<Scope> context, const variable_list& arguments)
        : Node(torch::autograd::collect_next_edges(arguments)), leaf(std::move(item)), scope(std::move(context)) {}
    variable_list apply(variable_list&& gradients) override;
};

struct EnterHook final : torch::autograd::FunctionPreHook {
    std::weak_ptr<Scope> scope;
    explicit EnterHook(std::weak_ptr<Scope> value) : scope(std::move(value)) {}
    variable_list operator()(const variable_list& values) override;
};

struct Scope : std::enable_shared_from_this<Scope> {
    variable_list parameters, sums;
    std::vector<ParameterState> states;
    std::unordered_map<std::string, std::shared_ptr<Leaf>> leaves;
    std::unique_ptr<ReplayPool> pool;
    // One-background-worker mode retains the original inline fold. Larger
    // budgets reserve one background thread for ordered reduction and disposal,
    // leaving the owning autograd thread free to produce independent replays.
    std::vector<std::shared_ptr<ReplayJob>> pending;
    size_t head = 0;
    std::deque<std::shared_ptr<ReplayJob>> ordered_pending;
    std::mutex reducer_mutex;
    std::condition_variable reducer_available, reducer_space, reducer_ready;
    std::thread reducer;
    const at::ThreadLocalState reducer_thread_state;
    bool reducer_initialized = false, reducer_stopping = false, producer_finished = false;
    std::exception_ptr reducer_error;
    int task = -1, workers;
    bool active = false, broken = false, closed = false;
    std::atomic<int64_t> occurrences{0};
    int64_t folded = 0;
    int64_t producer_backward_nanoseconds = 0, fold_nanoseconds = 0, future_wait_nanoseconds = 0;
    int64_t backpressure_nanoseconds = 0, max_pending_replays = 0;
    int64_t prewarm_nanoseconds = 0, prewarmed_leaves = 0;
    int64_t compacted_occurrences = 0, original_occurrence_edges = 0, retained_occurrence_edges = 0;

    void compact_deferred_edges(const Tensor& loss);

    Scope(variable_list values, int count) : parameters(std::move(values)), workers(count) {
        TORCH_CHECK(workers >= 0 && workers <= 255, "Native text replay supports 0 to 255 workers");
        TORCH_CHECK(at::get_num_threads() == 1, "Native text replay requires one ATen intra-op thread");
        std::unordered_set<c10::TensorImpl*> identities;
        for (const auto& parameter : parameters) {
            TORCH_CHECK(parameter.requires_grad() && parameter.is_leaf() && !parameter_hooks(parameter),
                        "Native text replay requires trainable leaf parameters without hooks");
            TORCH_CHECK(identities.insert(parameter.unsafeGetTensorImpl()).second,
                        "Native text replay requires distinct parameter identities");
            states.emplace_back(parameter);
        }
        if (workers) pool = std::make_unique<ReplayPool>(workers - (workers >= 2 ? 1 : 0));
        if (workers >= 2) {
            try {
                reducer = std::thread([this] { reduce(); });
                std::unique_lock<std::mutex> lock(reducer_mutex);
                reducer_ready.wait(lock, [&] { return reducer_initialized; });
                if (reducer_error) std::rethrow_exception(reducer_error);
            } catch (...) {
                stop_reducer();
                if (pool) pool->close();
                throw;
            }
        }
    }
    ~Scope() { close(false); }

    void check() const {
        TORCH_CHECK(!closed && !broken, "Native text replay scope is closed or failed");
        TORCH_CHECK(at::get_num_threads() == 1, "Native text replay thread count changed");
        for (size_t i = 0; i < parameters.size(); ++i)
            TORCH_CHECK(states[i].matches(parameters[i]) && !parameter_hooks(parameters[i]),
                        "Native text parameters or hooks changed inside a replay scope");
    }
    void check_readonly() const {
        TORCH_CHECK(!closed && !broken && !active, "Native prepared text requires an inactive open scope");
        TORCH_CHECK(at::get_num_threads() == 1, "Native text replay thread count changed");
        for (size_t i = 0; i < parameters.size(); ++i)
            TORCH_CHECK(states[i].matches(parameters[i]) && !parameter_hooks_readonly(parameters[i]),
                        "Native text parameters or hooks changed inside a replay scope");
    }
    std::shared_ptr<Leaf> prepared_leaf(const std::string& text, int64_t features) const {
        const auto found = leaves.find(text);
        TORCH_CHECK(found != leaves.end(), "Native text leaf was not prepared before parallel frame construction");
        found->second->check();
        TORCH_CHECK(features > 0 && found->second->value.dim() == 1 && found->second->value.numel() == features,
                    "Native prepared text feature size changed");
        return found->second;
    }
    void require_prepared(const std::vector<std::string>& texts, int64_t features) const {
        check_readonly();
        for (const auto& text : texts) if (!text.empty()) prepared_leaf(text, features);
    }
    void prewarm(const std::vector<std::string>& texts, int64_t features, int count) {
        check();
        TORCH_CHECK(!active && count >= 1 && count <= 256,
                    "Native text prewarm requires an inactive scope and 1 to 256 workers");
        WallTimer timer(prewarm_nanoseconds);
        std::unordered_set<std::string> seen;
        std::vector<std::string> missing;
        for (const auto& text : texts)
            if (!text.empty() && leaves.find(text) == leaves.end() && seen.insert(text).second)
                missing.push_back(text);
        TORCH_CHECK(leaves.size() + missing.size() <= 65536,
                    "Native text replay scope exceeded its unique leaf bound");
        if (missing.empty()) return;
        std::vector<std::shared_ptr<Leaf>> prepared(missing.size());
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        std::exception_ptr error;
        std::mutex error_mutex;
        const at::ThreadLocalState thread_state;
        auto run = [&] {
            try {
                at::get_num_threads();
                at::set_num_threads(1);
                TORCH_CHECK(at::get_num_threads() == 1, "Native text prewarm worker initialization failed");
                at::ThreadLocalStateGuard state_guard(thread_state);
                while (!failed.load()) {
                    const auto index = next.fetch_add(1);
                    if (index >= missing.size()) break;
                    prepared[index] = Leaf::make(parameters, missing[index], features);
                }
            } catch (...) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!error) error = std::current_exception();
            }
        };
        std::vector<std::thread> threads;
        try {
            const int total = std::min<int>(count, missing.size());
            for (int i = 1; i < total; ++i) threads.emplace_back(run);
            run();
        } catch (...) {
            failed.store(true);
            for (auto& thread : threads) if (thread.joinable()) thread.join();
            throw;
        }
        for (auto& thread : threads) thread.join();
        if (error) std::rethrow_exception(error);
        check();
        // Private alias graphs are independent. Outer occurrence priorities are
        // retained by serial construction or checked main-thread rebasing.
        for (size_t i = 0; i < missing.size(); ++i) leaves.emplace(std::move(missing[i]), std::move(prepared[i]));
        prewarmed_leaves += missing.size();
    }
    Tensor occurrence(const std::string& text, int64_t features) {
        check();
        auto found = leaves.find(text);
        if (found == leaves.end()) {
            TORCH_CHECK(leaves.size() < 65536, "Native text replay scope exceeded its unique leaf bound");
            found = leaves.emplace(text, Leaf::make(parameters, text, features)).first;
        }
        found->second->check();
        auto result = found->second->value.clone();
        auto node = std::make_shared<Occurrence>(found->second, shared_from_this());
        torch::autograd::set_history(result, node);
        occurrences.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    Tensor occurrence_prepared(const std::string& text, int64_t features, const variable_list& aliases) {
        check_readonly();
        const auto leaf = prepared_leaf(text, features);
        TORCH_CHECK(aliases.size() == parameters.size(), "Native prepared text alias count changed");
        std::unordered_set<c10::TensorImpl*> identities;
        variable_list arguments;
        arguments.reserve(leaf->arguments.size());
        for (size_t i = 0; i < aliases.size(); ++i) {
            const auto& alias = aliases[i];
            const auto& parameter = parameters[i];
            TORCH_CHECK(alias.defined() && alias.requires_grad() && alias.is_leaf() &&
                        std::none_of(parameters.begin(), parameters.end(),
                                     [&](const Tensor& canonical) { return alias.is_same(canonical); }) &&
                        identities.insert(alias.unsafeGetTensorImpl()).second &&
                        alias.storage().unsafeGetStorageImpl() == parameter.storage().unsafeGetStorageImpl() &&
                        alias.const_data_ptr() == parameter.const_data_ptr() && alias.scalar_type() == parameter.scalar_type() &&
                        alias.sizes() == parameter.sizes() && alias.strides() == parameter.strides() &&
                        alias.unsafeGetTensorImpl()->version_counter().current_version() == states[i].version &&
                        !parameter_hooks(alias),
                        "Native prepared text requires distinct private aliases of its canonical parameters");
            for (size_t j = 0; j < leaf->counts[i]; ++j) arguments.push_back(alias);
        }
        auto result = leaf->value.clone();
        auto node = std::make_shared<Occurrence>(leaf, shared_from_this(), arguments);
        torch::autograd::set_history(result, node);
        occurrences.fetch_add(1, std::memory_order_relaxed);
        return result;
    }
    bool defers() const {
        return active && task >= 0 && torch::autograd::get_current_graph_task_id() == task && at::get_num_threads() == 1;
    }
    void fold_job(const std::shared_ptr<ReplayJob>& job, variable_list values) {
        WallTimer timer(fold_nanoseconds);
        size_t cursor = 0;
        at::NoGradGuard no_grad;
        for (size_t i = 0; i < parameters.size(); ++i) {
            TORCH_CHECK(parameters[i].is_same(job->leaf->parameters[i]), "Native text parameters changed during fold");
            for (size_t j = 0; j < job->leaf->counts[i]; ++j) {
                const auto& value = values[cursor++];
                if (!value.defined()) continue;
                if (!sums[i].defined()) sums[i] = at::empty_like(parameters[i]).copy_(value);
                else sums[i].add_(value);
                ++folded;
            }
        }
        TORCH_CHECK(cursor == values.size(), "Native text replay produced extra contributions");
        // Releasing these small ATen views is part of reduction work. In async
        // mode it must stay off the producer thread as well as the additions.
        values.clear();
    }
    void fold_head() {
        auto job = pending[head];
        variable_list values;
        {
            WallTimer timer(future_wait_nanoseconds);
            values = job->future.get();
        }
        fold_job(job, std::move(values));
        pending[head++].reset();
    }
    void reduce() {
        std::optional<at::ThreadLocalStateGuard> state_guard;
        try {
            at::get_num_threads();
            at::set_num_threads(1);
            TORCH_CHECK(at::get_num_threads() == 1, "Native reducer initialization failed");
            state_guard.emplace(reducer_thread_state);
            {
                std::lock_guard<std::mutex> lock(reducer_mutex);
                reducer_initialized = true;
                reducer_ready.notify_all();
            }
            for (;;) {
                std::shared_ptr<ReplayJob> job;
                {
                    std::unique_lock<std::mutex> lock(reducer_mutex);
                    reducer_available.wait(lock, [&] {
                        return reducer_stopping || !ordered_pending.empty() || producer_finished;
                    });
                    if (reducer_stopping || (ordered_pending.empty() && producer_finished)) return;
                    job = ordered_pending.front();
                }
                variable_list values;
                {
                    WallTimer timer(future_wait_nanoseconds);
                    // A cleanup stop must not wait forever for a job that the
                    // replay pool has not claimed. Polling sleeps on the future.
                    while (job->future.wait_for(std::chrono::milliseconds(1)) != std::future_status::ready) {
                        std::lock_guard<std::mutex> lock(reducer_mutex);
                        if (reducer_stopping) return;
                    }
                    values = job->future.get();
                }
                fold_job(job, std::move(values));
                {
                    std::lock_guard<std::mutex> lock(reducer_mutex);
                    ordered_pending.pop_front();
                }
                reducer_space.notify_one();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(reducer_mutex);
            reducer_error = std::current_exception();
            reducer_initialized = true;
            reducer_ready.notify_all();
            reducer_space.notify_all();
        }
    }
    void stop_reducer() {
        {
            std::lock_guard<std::mutex> lock(reducer_mutex);
            reducer_stopping = true;
        }
        reducer_available.notify_all();
        reducer_space.notify_all();
        if (reducer.joinable()) reducer.join();
    }
    void add(const std::shared_ptr<Leaf>& leaf, const Tensor& gradient) {
        TORCH_CHECK(pool && defers(), "Native text replay was deferred outside its own backward");
        if (workers >= 2) {
            auto job = std::make_shared<ReplayJob>(leaf, gradient);
            {
                std::unique_lock<std::mutex> lock(reducer_mutex);
                if (ordered_pending.size() >= 1024) {
                    WallTimer timer(backpressure_nanoseconds);
                    reducer_space.wait(lock, [&] {
                        return reducer_stopping || reducer_error || ordered_pending.size() < 1024;
                    });
                }
                if (reducer_error) std::rethrow_exception(reducer_error);
                TORCH_CHECK(!reducer_stopping && !producer_finished, "Native ordered reducer is closed");
                ordered_pending.push_back(job);
                max_pending_replays = std::max<int64_t>(max_pending_replays, ordered_pending.size());
            }
            pool->submit(job);
            reducer_available.notify_one();
            return;
        }
        // Bound queued gradient storage independently of the number of occurrences.
        if (pending.size() - head >= 1024) fold_head();
        auto job = std::make_shared<ReplayJob>(leaf, gradient);
        pending.push_back(job);
        max_pending_replays = std::max<int64_t>(max_pending_replays, pending.size() - head);
        pool->submit(job);
        while (head < pending.size() && pending[head]->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            fold_head();
    }
    void backward(const Tensor& loss) {
        check();
        TORCH_CHECK(!active && pending.empty() && !producer_finished,
                    "Native text replay backward is already active or completed");
        TORCH_CHECK(loss.defined() && loss.numel() == 1 && loss.grad_fn(), "Native text replay requires a scalar differentiable loss");
        if (!pool) {
            WallTimer timer(producer_backward_nanoseconds);
            torch::autograd::backward({loss});
            return;
        }
        for (const auto& parameter : parameters)
            TORCH_CHECK(!parameter.grad().defined(), "Native text replay backward requires cleared parameter gradients");
        sums.resize(parameters.size());
        compact_deferred_edges(loss);
        active = true;
        task = -1;
        Tensor root;
        {
            at::AutoGradMode enable(true);
            root = loss.view_as(loss);
            root.grad_fn()->add_pre_hook(std::make_unique<EnterHook>(shared_from_this()));
        }
        try {
            {
                WallTimer timer(producer_backward_nanoseconds);
                torch::autograd::backward({root});
            }
            if (workers >= 2) {
                {
                    std::lock_guard<std::mutex> lock(reducer_mutex);
                    producer_finished = true;
                }
                reducer_available.notify_all();
                if (reducer.joinable()) reducer.join();
                if (reducer_error) std::rethrow_exception(reducer_error);
                TORCH_CHECK(ordered_pending.empty(), "Native ordered reducer left unfinished work");
            }
            size_t cursor = head;
            while (head < pending.size()) {
                if (pending[head]->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                    cursor = std::max(cursor, head);
                    while (cursor < pending.size() && !pool->run(pending[cursor], true)) ++cursor;
                    if (cursor < pending.size()) { ++cursor; continue; }
                }
                fold_head();
            }
            check();
            for (const auto& parameter : parameters)
                TORCH_CHECK(!parameter.grad().defined(), "Native text parameter received gradient outside its replayed occurrences");
            // Validate the entire set before installing any sum.
            for (size_t i = 0; i < parameters.size(); ++i) if (sums[i].defined()) parameters[i].mutable_grad() = sums[i];
            active = false;
            task = -1;
            pending.clear();
            sums.clear();
            head = 0;
        } catch (...) {
            broken = true;
            active = false;
            task = -1;
            throw;
        }
    }
    void close(bool reject_unfinished = true) {
        if (closed) return;
        stop_reducer();
        const bool unfinished = active || !pending.empty() || !ordered_pending.empty();
        if (pool) pool->close();
        for (auto& [_, leaf] : leaves) leaf->release();
        pending.clear(); ordered_pending.clear(); sums.clear();
        active = false; closed = true;
        TORCH_CHECK(!reject_unfinished || !unfinished, "Native text replay scope closed with unfinished backward work");
    }
    std::map<std::string, int64_t> statistics() const {
        auto result = pool ? pool->statistics() : std::map<std::string, int64_t>{};
        int64_t tape_leaf_count = 0, tape_replay_count = 0, engine_replay_count = 0;
        for (const auto& [_, leaf] : leaves) {
            tape_leaf_count += leaf->tape_fallback_reason.empty();
            tape_replay_count += leaf->tape_replays.load(std::memory_order_relaxed);
            engine_replay_count += leaf->engine_replays.load(std::memory_order_relaxed);
        }
        result.insert({{"workers", workers}, {"unique_texts", static_cast<int64_t>(leaves.size())},
                       {"occurrences", occurrences.load(std::memory_order_relaxed)}, {"folded_contributions", folded},
                       {"tape_leaves", tape_leaf_count}, {"tape_replays", tape_replay_count},
                       {"engine_replays", engine_replay_count},
                       {"reducer_threads", workers >= 2 ? 1 : 0},
                       {"replay_threads", workers - (workers >= 2 ? 1 : 0)},
                       {"producer_backward_nanoseconds", producer_backward_nanoseconds},
                       {"fold_nanoseconds", fold_nanoseconds}, {"future_wait_nanoseconds", future_wait_nanoseconds},
                       {"backpressure_nanoseconds", backpressure_nanoseconds}, {"max_pending_replays", max_pending_replays},
                       {"prewarm_nanoseconds", prewarm_nanoseconds}, {"prewarmed_leaves", prewarmed_leaves}});
        result.insert({{"compacted_occurrences", compacted_occurrences},
                       {"original_occurrence_edges", original_occurrence_edges},
                       {"retained_occurrence_edges", retained_occurrence_edges}});
        return result;
    }
};

inline void Scope::compact_deferred_edges(const Tensor& loss) {
    if (!pool || !BuiltinReplayTape::ordinary_execution_context()) return;
    using namespace torch::autograd::generated;
    // Unknown/custom graphs and observable hooks retain the original Engine
    // graph. This optimization is restricted to the ordinary native PPO graph.
    // A null inspector means the exact listed type has no SavedVariable fields
    // in the ABI-bound ATen headers. Non-null inspectors check every such field:
    // unpack hooks survive after their forward-time TLS context has ended.
    using SavedInspector = bool (*)(const Node&);
    static const std::unordered_map<std::type_index, SavedInspector> allowed{
        {typeid(AddBackward0), nullptr}, {typeid(SubBackward0), nullptr},
        {typeid(MulBackward0), [](const Node& node) {
            const auto& value = static_cast<const MulBackward0&>(node);
            return !value.self_.has_hooks() && !value.other_.has_hooks(); }},
        {typeid(MulBackward1), nullptr},
        {typeid(DivBackward0), [](const Node& node) {
            const auto& value = static_cast<const DivBackward0&>(node);
            return !value.self_.has_hooks() && !value.other_.has_hooks(); }},
        {typeid(DivBackward1), nullptr},
        {typeid(PowBackward0), [](const Node& node) {
            return !static_cast<const PowBackward0&>(node).self_.has_hooks(); }},
        {typeid(ExpBackward0), [](const Node& node) {
            return !static_cast<const ExpBackward0&>(node).result_.has_hooks(); }},
        {typeid(ClampBackward1), [](const Node& node) {
            return !static_cast<const ClampBackward1&>(node).self_.has_hooks(); }},
        {typeid(MinimumBackward0), [](const Node& node) {
            const auto& value = static_cast<const MinimumBackward0&>(node);
            return !value.self_.has_hooks() && !value.other_.has_hooks(); }},
        {typeid(MaximumBackward0), [](const Node& node) {
            const auto& value = static_cast<const MaximumBackward0&>(node);
            return !value.self_.has_hooks() && !value.other_.has_hooks(); }},
        {typeid(NegBackward0), nullptr}, {typeid(MeanBackward0), nullptr},
        {typeid(MeanBackward1), nullptr}, {typeid(SumBackward0), nullptr}, {typeid(SumBackward1), nullptr},
        {typeid(StackBackward0), nullptr}, {typeid(SelectBackward0), nullptr}, {typeid(ViewBackward0), nullptr},
        {typeid(UnsafeViewBackward0), nullptr}, {typeid(UnsqueezeBackward0), nullptr},
        {typeid(LogSoftmaxBackward0), [](const Node& node) {
            return !static_cast<const LogSoftmaxBackward0&>(node).result_.has_hooks(); }},
        {typeid(SqueezeBackward1), nullptr},
        {typeid(MmBackward0), [](const Node& node) {
            const auto& value = static_cast<const MmBackward0&>(node);
            return !value.self_.has_hooks() && !value.mat2_.has_hooks(); }},
        {typeid(MvBackward0), [](const Node& node) {
            const auto& value = static_cast<const MvBackward0&>(node);
            return !value.self_.has_hooks() && !value.vec_.has_hooks(); }},
        {typeid(AddmmBackward0), [](const Node& node) {
            const auto& value = static_cast<const AddmmBackward0&>(node);
            return !value.mat1_.has_hooks() && !value.mat2_.has_hooks(); }},
        {typeid(TBackward0), nullptr}, {typeid(CatBackward0), nullptr},
        {typeid(SigmoidBackward0), [](const Node& node) {
            return !static_cast<const SigmoidBackward0&>(node).result_.has_hooks(); }},
        {typeid(TanhBackward0), [](const Node& node) {
            return !static_cast<const TanhBackward0&>(node).result_.has_hooks(); }},
        {typeid(UnbindBackward0), nullptr}, {typeid(UnsafeSplitBackward0), nullptr},
        {typeid(TransposeBackward0), nullptr},
        {typeid(EmbeddingBackward0), [](const Node& node) {
            return !static_cast<const EmbeddingBackward0&>(node).indices_.has_hooks(); }},
        {typeid(ConvolutionBackward0), [](const Node& node) {
            const auto& value = static_cast<const ConvolutionBackward0&>(node);
            return !value.input_.has_hooks() && !value.weight_.has_hooks(); }},
        {typeid(IndexAddBackward0), [](const Node& node) {
            const auto& value = static_cast<const IndexAddBackward0&>(node);
            return !value.index_.has_hooks() && !value.source_.has_hooks(); }},
        {typeid(AdaptiveAvgPool2DBackward0), [](const Node& node) {
            return !static_cast<const AdaptiveAvgPool2DBackward0&>(node).self_.has_hooks(); }},
        {typeid(PermuteBackward0), nullptr}, {typeid(SliceBackward0), nullptr}, {typeid(ExpandBackward0), nullptr},
        {typeid(IndexSelectBackward0), [](const Node& node) {
            return !static_cast<const IndexSelectBackward0&>(node).index_.has_hooks(); }},
        {typeid(CloneBackward0), nullptr}, {typeid(ReshapeAliasBackward0), nullptr},
        {typeid(torch::autograd::AccumulateGrad), nullptr}, {typeid(Occurrence), nullptr}};
    struct Replacement {
        std::shared_ptr<Occurrence> occurrence;
        torch::autograd::edge_list edges;
    };
    std::vector<Replacement> replacements;
    std::vector<std::shared_ptr<Node>> pending{loss.grad_fn()};
    std::unordered_set<Node*> seen{loss.grad_fn().get()};
    while (!pending.empty()) {
        auto node = std::move(pending.back()); pending.pop_back();
        const auto& function = *node;
        const auto inspector = allowed.find(typeid(function));
        if (inspector == allowed.end() || (inspector->second && !inspector->second(function)) ||
            !node->pre_hooks().empty() ||
            !node->tensor_pre_hooks().empty() || !node->retains_grad_hooks().empty() ||
            !node->post_hooks().empty() || seen.size() > 1000000) return;
        if (typeid(function) == typeid(torch::autograd::AccumulateGrad) &&
            parameter_hooks(static_cast<const torch::autograd::AccumulateGrad&>(function).variable)) return;
        if (typeid(function) == typeid(Occurrence)) {
            auto occurrence = std::static_pointer_cast<Occurrence>(node);
            if (occurrence->scope.get() != this || !occurrence->leaf || occurrence->compact_edges) return;
            const auto& leaf = *occurrence->leaf;
            // The leaf's private graph is outside this outer traversal. A
            // compiled tape proves that it has no external saved-tensor hooks.
            if (!leaf.tape) return;
            const auto& edges = node->next_edges();
            if (leaf.parameters.size() != parameters.size() || edges.size() != leaf.arguments.size()) return;
            torch::autograd::edge_list unique;
            size_t cursor = 0;
            for (size_t i = 0; i < parameters.size(); ++i) {
                if (!parameters[i].is_same(leaf.parameters[i])) return;
                const auto accumulator = torch::autograd::impl::try_get_grad_accumulator(parameters[i]);
                const auto* terminal = accumulator.get();
                if (!terminal || typeid(*terminal) != typeid(torch::autograd::AccumulateGrad)) return;
                for (size_t j = 0; j < leaf.counts[i]; ++j) {
                    if (cursor >= edges.size() || edges[cursor].function != accumulator || edges[cursor].input_nr != 0)
                        return;
                    if (j == 0) unique.push_back(edges[cursor]);
                    ++cursor;
                }
            }
            if (cursor != edges.size() || unique.empty()) return;
            if (unique.size() < edges.size()) replacements.push_back({std::move(occurrence), std::move(unique)});
        }
        for (const auto& edge : node->next_edges())
            if (edge.is_valid() && seen.insert(edge.function.get()).second) pending.push_back(edge.function);
    }
    // Every original contribution is still computed and folded by the same
    // ordered reducer. The outer Engine only receives undefined gradients for
    // these edges. Duplicate edges all arrive together from one Occurrence, so
    // retaining one per parameter preserves each terminal's readiness event.
    // Nonterminal dependencies, sequence numbers and numerical calls are intact.
    // Validate the complete graph before replacing any edge list.
    for (auto& replacement : replacements) {
        auto& occurrence = *replacement.occurrence;
        original_occurrence_edges += occurrence.next_edges().size();
        retained_occurrence_edges += replacement.edges.size();
        occurrence.compact_edges = replacement.edges.size();
        occurrence.next_edges() = std::move(replacement.edges);
        ++compacted_occurrences;
    }
}

inline variable_list Occurrence::apply(variable_list&& gradients) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(!at::GradMode::is_enabled(), "Native replayed text leaves do not support higher-order gradients");
    TORCH_CHECK(leaf && scope, "A native text occurrence supports a single backward");
    auto saved_leaf = std::move(leaf);
    auto saved_scope = std::move(scope);
    TORCH_CHECK(!saved_scope->closed && !saved_scope->broken, "Native text occurrence belongs to a released scope");
    TORCH_CHECK(gradients.size() == 1, "Native text occurrence expects one output gradient");
    TORCH_CHECK(!compact_edges || saved_scope->defers(),
                "Compacted native text edges require their owned deferred backward");
    const auto output_edges = compact_edges ? compact_edges : saved_leaf->arguments.size();
    if (!gradients[0].defined()) return variable_list(output_edges);
    if (saved_scope->defers()) {
        saved_scope->add(saved_leaf, gradients[0]);
        return variable_list(output_edges);
    }
    return saved_leaf->replay(gradients[0]);
}

inline variable_list EnterHook::operator()(const variable_list& values) {
    auto owner = scope.lock();
    TORCH_CHECK(owner && owner->active, "Native text replay root belongs to an inactive scope");
    if (owner->task < 0) owner->task = torch::autograd::get_current_graph_task_id();
    return values;
}
}  // namespace native_text
