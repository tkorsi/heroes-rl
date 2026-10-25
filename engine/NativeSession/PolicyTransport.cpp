#include "StdInc.h"
#include "PolicyTransport.h"
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>

VCMI_LIB_NAMESPACE_BEGIN
namespace native_session
{
class InProcessPolicy::Impl
{
public:
	std::mutex mutex;
	std::condition_variable changed;
	const uint64_t generation;
	const size_t capacity;
	uint64_t decision = 0;
	bool closed = false;
	bool terminal = false;
	bool waiting = false;
	std::string failure;
	std::vector<JsonNode> messages;
	std::optional<Boundary> boundary;
	std::optional<JsonNode> response;
	explicit Impl(uint64_t generation_, size_t capacity_) : generation(generation_), capacity(capacity_) {}
	void check() const
	{
		if(!failure.empty()) throw std::runtime_error(failure);
		if(closed) throw std::runtime_error("Native policy channel closed");
	}
};

InProcessPolicy::InProcessPolicy(uint64_t generation, size_t capacity)
	: impl(std::make_unique<Impl>(generation, capacity))
{
	if(!generation || !capacity) throw std::invalid_argument("Channel needs a nonzero generation/capacity");
}
InProcessPolicy::~InProcessPolicy() = default;

WriteMetrics InProcessPolicy::send(const JsonNode & message)
{
	const auto start = std::chrono::steady_clock::now();
	std::lock_guard lock(impl->mutex);
	impl->check();
	if(impl->terminal) throw std::runtime_error("Message after terminal");
	if(impl->messages.size() >= impl->capacity)
	{
		impl->failure = "Native policy message queue overflow";
		impl->changed.notify_all();
		throw std::runtime_error(impl->failure);
	}
	// Never block a packet callback while it owns the event/game-state locks.
	impl->messages.push_back(message);
	impl->terminal = message["type"].String() == "terminal";
	if(impl->terminal)
	{
		// Terminal invalidates a held decision without redelivering its already
		// observed frames. Its new token can never authorize an action.
		++impl->decision;
		impl->boundary.reset();
	}
	if(message["type"].String() == "error")
		impl->failure = "ExplorerRL: " + message["message"].String();
	impl->changed.notify_all();
	return {0, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count(), 0};
}

JsonNode InProcessPolicy::receive(Deadline deadline)
{
	std::unique_lock lock(impl->mutex);
	impl->check();
	if(impl->terminal || impl->waiting || impl->response)
		throw std::runtime_error("Overlapping native policy receive");
	impl->waiting = true;
	++impl->decision;
	impl->changed.notify_all();
	if(!impl->changed.wait_until(lock, deadline, [&] { return impl->closed || !impl->failure.empty() || impl->response.has_value(); }))
	{
		impl->failure = "Native policy response timeout";
		impl->waiting = false;
		impl->changed.notify_all();
		throw std::runtime_error(impl->failure);
	}
	impl->check();
	auto result = std::move(*impl->response);
	impl->response.reset();
	impl->waiting = false;
	return result;
}

Boundary InProcessPolicy::observe(Deadline deadline)
{
	std::unique_lock lock(impl->mutex);
	if(!impl->changed.wait_until(lock, deadline, [&] {
		return impl->closed || !impl->failure.empty() || impl->boundary || impl->waiting || impl->terminal;
	})) throw std::runtime_error("Native observe timeout");
	impl->check();
	if(!impl->boundary)
	{
		impl->boundary = Boundary{{impl->generation, impl->decision}, std::move(impl->messages), impl->waiting && !impl->terminal, impl->terminal};
		impl->messages.clear();
	}
	return *impl->boundary; // observe is idempotent until a successful submit
}

void InProcessPolicy::submit(BoundaryToken token, const JsonNode & response)
{
	// Preserve the original socket response-size admission even without a wire.
	if(response.toCompactString().size() > 65536)
		throw std::invalid_argument("Policy response exceeds 64 KiB");
	std::lock_guard lock(impl->mutex);
	impl->check();
	if(impl->terminal || !impl->boundary || !impl->boundary->needsAction || !impl->waiting || impl->response
		|| token != impl->boundary->token || token.generation != impl->generation)
		throw std::invalid_argument("Stale, duplicate or nondecision native step token");
	impl->response = response;
	impl->boundary.reset();
	// A subsequent observe must wait for the NEXT receive, not the still
	// waking previous one. receive() owns clearing waiting after response.
	impl->waiting = false;
	impl->changed.notify_all();
}

void InProcessPolicy::close() noexcept
{
	std::lock_guard lock(impl->mutex);
	impl->closed = true;
	impl->changed.notify_all();
}
bool InProcessPolicy::cancelled() const noexcept
{
	std::lock_guard lock(impl->mutex);
	return impl->closed;
}
void InProcessPolicy::fail(const std::string & reason) noexcept
{
	std::lock_guard lock(impl->mutex);
	impl->failure = reason;
	impl->changed.notify_all();
}

namespace
{
struct Binding { std::string episode; std::shared_ptr<PolicyTransport> policy; bool claimed = false; };
std::mutex bindingsMutex;
std::map<int, Binding> bindings;
}
void installPolicy(int player, const std::string & episode, std::shared_ptr<PolicyTransport> policy)
{
	std::lock_guard lock(bindingsMutex);
	if(player < 0 || player > 7 || episode.empty() || !policy || bindings.contains(player))
		throw std::invalid_argument("Invalid/duplicate native policy binding");
	bindings.emplace(player, Binding{episode, std::move(policy)});
}
std::shared_ptr<PolicyTransport> claimPolicy(int player, const std::string & episode)
{
	std::lock_guard lock(bindingsMutex);
	const auto found = bindings.find(player);
	if(found == bindings.end()) return {};
	if(found->second.episode != episode || found->second.claimed)
		throw std::runtime_error("Native policy episode/player binding mismatch or duplicate claim");
	found->second.claimed = true;
	return found->second.policy;
}
void removePolicy(int player) noexcept
{
	std::lock_guard lock(bindingsMutex);
	const auto found = bindings.find(player);
	if(found != bindings.end()) found->second.policy->close();
	bindings.erase(player);
}
}
VCMI_LIB_NAMESPACE_END
