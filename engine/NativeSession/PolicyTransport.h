/* Native session transport seam. GPL-2.0-or-later. */
#pragma once
#include "../Global.h"
#include "../lib/json/JsonNode.h"
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
namespace native_session
{
using Deadline = std::chrono::steady_clock::time_point;
/// Only an explicit, failure-free channel close has cancellation semantics.
class DLL_LINKAGE PolicyTransportCancelled : public std::runtime_error
{
public:
	PolicyTransportCancelled() : std::runtime_error("Native policy channel closed") {}
	explicit PolicyTransportCancelled(const char * reason) : std::runtime_error(reason) {}
};
/// The player's own terminal superseded a pending receive. Like a closed
/// socket after the natural end, this is never a failure or an action.
class DLL_LINKAGE PolicyTransportTerminal final : public PolicyTransportCancelled
{
public:
	PolicyTransportTerminal() : PolicyTransportCancelled("Native policy terminal superseded the receive") {}
};
struct WriteMetrics
{
	int64_t encodeMicros = 0;
	int64_t sendMicros = 0;
	int64_t wireBytes = 0; ///< zero for the typed, nonserialized channel
};

/// Only already-sanitized ExplorerRL messages cross this interface. It does
/// not expose CGameState, callbacks, engine IDs outside existing messages,
/// RNG or authoritative replay evidence. close() must wake receive().
class DLL_LINKAGE PolicyTransport
{
public:
	virtual ~PolicyTransport() = default;
	virtual WriteMetrics send(const JsonNode & message) = 0;
	virtual JsonNode receive(Deadline deadline) = 0;
	virtual void close() noexcept = 0;
	virtual bool cancelled() const noexcept { return false; }
};

struct BoundaryToken
{
	uint64_t generation = 0;
	uint64_t decision = 0;
	bool operator==(const BoundaryToken &) const = default;
};
struct Boundary
{
	BoundaryToken token;
	std::vector<JsonNode> messages; ///< ordered, unmodified player-visible stream
	bool needsAction = false;
	bool terminalObserved = false; ///< not replay verification or a scored result
};
/// Incremental delivery: every message is handed out exactly once, in order,
/// including a failure prefix and messages after the player's own terminal.
struct Delivery
{
	BoundaryToken token;
	std::vector<JsonNode> messages;
	bool needsAction = false; ///< newly offered decision; token may authorize exactly one submit
	bool terminalObserved = false; ///< state: the player's own terminal was sent
	bool closed = false;
	bool timedOut = false; ///< deadline passed with nothing new
	std::string failure; ///< first failure reason; messages are its delivered prefix
};

/// Per-episode, per-player channel. Reuse creates a NEW object/generation;
/// old handles remain closed forever. No OS descriptors or process assumptions.
/// Use either observe() (idempotent boundary) or take() (incremental) per channel.
class DLL_LINKAGE InProcessPolicy final : public PolicyTransport
{
	class Impl;
	std::unique_ptr<Impl> impl;
public:
	explicit InProcessPolicy(uint64_t generation, size_t maxQueuedMessages = 64);
	~InProcessPolicy() override;
	WriteMetrics send(const JsonNode & message) override;
	JsonNode receive(Deadline deadline) override;
	void close() noexcept override;
	bool cancelled() const noexcept override;
	bool failed() const noexcept;
	std::string failure() const;
	Boundary observe(Deadline deadline);
	/// Waits until a decision, terminal, failure or close is newly available
	/// (or, unless untilBoundary, any message; or the queue is half full).
	/// Never throws on channel failure.
	Delivery take(Deadline deadline, bool untilBoundary);
	/// Non-blocking: messages not yet handed out, e.g. after all producers joined.
	std::vector<JsonNode> drain();
	/// lineAdmitted: the caller already applied the socket's 64 KiB limit to
	/// the response's own wire line; otherwise its JsonWriter text is bounded.
	void submit(BoundaryToken token, const JsonNode & response, bool lineAdmitted = false);
	void fail(const std::string & reason) noexcept;
};

/// Shared-library registry so the executable and dynamically loaded ExplorerRL
/// use the same endpoint. Install/remove only with all previous episode threads
/// joined. A registered endpoint may be claimed exactly once by its player.
DLL_LINKAGE void installPolicy(int player, const std::string & episode, std::shared_ptr<PolicyTransport> policy);
DLL_LINKAGE std::shared_ptr<PolicyTransport> claimPolicy(int player, const std::string & episode);
DLL_LINKAGE void removePolicy(int player) noexcept;
}
VCMI_LIB_NAMESPACE_END
