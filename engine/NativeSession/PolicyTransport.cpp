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
	bool terminalReported = false;
	bool waiting = false;
	std::string failure; ///< the first reason only; later reports never replace it
	std::vector<JsonNode> messages; ///< not yet handed to the host
	std::optional<Boundary> boundary; ///< idempotent observe() view
	std::optional<uint64_t> offered; ///< decision handed out as needing an action
	std::optional<uint64_t> superseded; ///< offered decision the player's own terminal invalidated
	std::optional<JsonNode> response;
	explicit Impl(uint64_t generation_, size_t capacity_) : generation(generation_), capacity(capacity_) {}
	void check() const
	{
		if(!failure.empty()) throw std::runtime_error(failure);
		if(closed) throw PolicyTransportCancelled();
	}
	void setFailure(const std::string & reason)
	{
		if(failure.empty()) failure = reason;
	}
	bool actionPending() const
	{
		return waiting && !terminal && !response && failure.empty() && !closed;
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
	// Like a socket peer that stopped reading: a closed channel refuses. A
	// failed but open channel still accepts the adapter's failure prefix and
	// its error record, so the host receives the same evidence a socket would.
	if(impl->closed) throw PolicyTransportCancelled();
	if(impl->messages.size() >= impl->capacity)
	{
		impl->setFailure("Native policy message queue overflow");
		impl->changed.notify_all();
		throw std::runtime_error(impl->failure);
	}
	// Never block a packet callback while it owns the event/game-state locks.
	impl->messages.push_back(message);
	bool terminal = message["type"].String() == "terminal";
	if(message["type"].String() == "contextual" && message["records"].isVector())
		for(const auto & record : message["records"].Vector())
			if(record["kind"].String() == "terminal") terminal = true;
	if(terminal && !impl->terminal)
	{
		// Terminal invalidates a held decision without redelivering its already
		// observed frames. Its new token can never authorize an action. Later
		// permitted final envelopes are queued behind it, never rejected.
		impl->terminal = true;
		if(impl->offered == impl->decision) impl->superseded = impl->offered;
		++impl->decision;
		impl->boundary.reset();
		impl->offered.reset();
	}
	if(message["type"].String() == "error")
		impl->setFailure("ExplorerRL: " + message["message"].String());
	impl->changed.notify_all();
	return {0, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count(), 0};
}

JsonNode InProcessPolicy::receive(Deadline deadline)
{
	std::unique_lock lock(impl->mutex);
	impl->check();
	if(impl->terminal) throw PolicyTransportTerminal();
	if(impl->waiting || impl->response)
		throw std::runtime_error("Overlapping native policy receive");
	impl->waiting = true;
	++impl->decision;
	impl->changed.notify_all();
	if(!impl->changed.wait_until(lock, deadline, [&] {
		return impl->closed || !impl->failure.empty() || impl->response.has_value() || impl->terminal;
	}))
	{
		impl->setFailure("Native policy response timeout");
		impl->waiting = false;
		impl->changed.notify_all();
		throw std::runtime_error(impl->failure);
	}
	impl->check();
	if(!impl->response)
	{
		impl->waiting = false;
		throw PolicyTransportTerminal();
	}
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
		const bool action = impl->actionPending();
		impl->boundary = Boundary{{impl->generation, impl->decision}, std::move(impl->messages), action, impl->terminal};
		impl->messages.clear();
		if(action) impl->offered = impl->decision;
		if(impl->terminal) impl->terminalReported = true;
	}
	return *impl->boundary; // observe is idempotent until a successful submit
}

Delivery InProcessPolicy::take(Deadline deadline, bool untilBoundary)
{
	std::unique_lock lock(impl->mutex);
	if(impl->boundary) throw std::logic_error("Native channel already uses idempotent observe()");
	const bool woke = impl->changed.wait_until(lock, deadline, [&] {
		return impl->closed || !impl->failure.empty() || (impl->terminal && !impl->terminalReported)
			|| (impl->actionPending() && impl->offered != impl->decision) || (!untilBoundary && !impl->messages.empty())
			// High water: a long opponent turn is handed over before send() could overflow.
			|| impl->messages.size() * 2 >= impl->capacity;
	});
	Delivery result;
	result.token = {impl->generation, impl->decision};
	result.messages = std::move(impl->messages);
	impl->messages.clear();
	// A decision is offered once; terminal, close and failure are reported as states.
	const bool pending = impl->actionPending();
	result.needsAction = pending && impl->offered != impl->decision;
	if(pending) impl->offered = impl->decision;
	const bool newTerminal = impl->terminal && !impl->terminalReported;
	result.terminalObserved = impl->terminal;
	impl->terminalReported = impl->terminal;
	result.closed = impl->closed;
	result.failure = impl->failure;
	result.timedOut = !woke && result.messages.empty() && !result.needsAction && !newTerminal
		&& !result.closed && result.failure.empty();
	return result;
}

std::vector<JsonNode> InProcessPolicy::drain()
{
	std::lock_guard lock(impl->mutex);
	auto result = std::move(impl->messages);
	impl->messages.clear();
	return result;
}

bool InProcessPolicy::submit(BoundaryToken token, const JsonNode & response, bool lineAdmitted)
{
	// Preserve the original socket response-size admission even without a wire.
	if(!lineAdmitted && response.toCompactString().size() > 65536)
		throw std::invalid_argument("Policy response exceeds 64 KiB");
	std::lock_guard lock(impl->mutex);
	impl->check();
	if(impl->terminal && impl->superseded && token == BoundaryToken{impl->generation, *impl->superseded})
	{
		// The socket backend writes this line to a peer that already sent its
		// terminal; the engine never reads it. Once only, never an action.
		impl->superseded.reset();
		return false;
	}
	if(impl->terminal || !impl->waiting || impl->response || impl->offered != impl->decision
		|| token != BoundaryToken{impl->generation, impl->decision})
		throw std::invalid_argument("Stale, duplicate or nondecision native step token");
	impl->response = response;
	impl->boundary.reset();
	impl->offered.reset();
	// A subsequent observe must wait for the NEXT receive, not the still
	// waking previous one. receive() owns clearing waiting after response.
	impl->waiting = false;
	impl->changed.notify_all();
	return true;
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
bool InProcessPolicy::failed() const noexcept
{
	std::lock_guard lock(impl->mutex);
	return !impl->failure.empty();
}
std::string InProcessPolicy::failure() const
{
	std::lock_guard lock(impl->mutex);
	return impl->failure;
}
void InProcessPolicy::fail(const std::string & reason) noexcept
{
	std::lock_guard lock(impl->mutex);
	try { impl->setFailure(reason.empty() ? std::string("Native policy failure") : reason); }
	catch(...) { if(impl->failure.empty()) impl->failure = "Native policy failure"; }
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
