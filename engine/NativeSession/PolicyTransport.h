/* Native session transport seam. GPL-2.0-or-later. */
#pragma once
#include "../Global.h"
#include "../lib/json/JsonNode.h"
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
namespace native_session
{
using Deadline = std::chrono::steady_clock::time_point;
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

/// Per-episode, per-player channel. Reuse creates a NEW object/generation;
/// old handles remain closed forever. No OS descriptors or process assumptions.
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
	Boundary observe(Deadline deadline);
	void submit(BoundaryToken token, const JsonNode & response);
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
