/* Experimental live VCMI session. GPL-2.0-or-later. */
#pragma once
#include "PolicyTransport.h"
#include "Opponent.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
namespace native_session
{
struct SessionOptions
{
	bool consoleLog = true; ///< mirror the engine log to stdout, as the executable client does
};
struct EpisodeConfig
{
	std::string episode;
	std::string map; ///< existing VCMI resource URI, as used by --testmap
	std::string captureDirectory; ///< new, absolute directory; never reused
	int seed = 0;
	int learner = 0; ///< other side is the unchanged Nullkiller Easy
	int policyTimeoutSeconds = 30;
	std::string logPath; ///< optional new, absolute per-game engine log in console format
	std::string interface = "adventure-dialogs-v1"; ///< ExplorerRL player interface; ExplorerRL validates the name
};
struct SessionTimings
{
	double loadSeconds = 0;
	double resetSeconds = 0; ///< startup dispatch plus waits through first held boundary; caller idle excluded
	double advanceSeconds = 0; ///< host waits for the engine after a response until the next boundary
	double cleanupSeconds = 0;
	double finishWaitSeconds = 0; ///< natural terminal barrier waits after the own terminal
	uint64_t episodes = 0;
	uint64_t steps = 0;
	uint64_t opponentTurns = 0;
	double opponentTurnSeconds = 0; ///< built-in opponent turns, engine work included (nested in advance)
	uint64_t battles = 0;
	double battleSeconds = 0; ///< automatic battles, both BattleAI sides and engine work (may overlap opponent turns)
};
/// What finish() proved about one game. Never a replay verification.
struct EpisodeReport
{
	std::optional<int> exitCode; ///< native headless status; 0 only after the natural terminal barrier
	bool natural = false; ///< finish followed the player's own terminal
	bool hostStopped = false; ///< the host stopped the game before any exit status existed
	bool shutdownDeadline = false; ///< a natural finish did not reach its exit status in time
	bool closeClean = false; ///< every episode thread joined and no failure was discovered
	bool poisoned = false; ///< this owner can never reset again
	bool experimentFailed = false;
	std::string channelFailure;
	std::string closeError;
	std::vector<JsonNode> messages; ///< undelivered player stream, read after all producers joined
	native_session_host::OpponentStats opponent;
	double waitSeconds = 0;
	double closeSeconds = 0;
};

/// Experimental, opt-in host of the existing headless game, not replay.
/// One owner per process because LIBRARY/GAME/settings remain global. Calls
/// from one controller thread only; the policy channel owns thread handoff.
/// Loaded resources are retained, not certified transitively immutable.
class EngineSession
{
	class Impl;
	std::unique_ptr<Impl> impl;
public:
	explicit EngineSession(SessionOptions options = {});
	~EngineSession();
	EngineSession(const EngineSession &) = delete;
	EngineSession & operator=(const EngineSession &) = delete;
	void load();
	void reset(const EpisodeConfig & config);
	Boundary observe(Deadline deadline);
	Boundary step(BoundaryToken token, const JsonNode & response, Deadline deadline);
	/// Incremental alternative to observe/step: each message is delivered once.
	Delivery take(Deadline deadline, bool untilBoundary = true);
	/// False: the player's own terminal superseded this decision; admitted, never applied.
	bool submit(BoundaryToken token, const JsonNode & response, bool lineAdmitted = false);
	/// Ends the game: natural waits for the terminal barrier's exit status
	/// until deadline, then (always) joins every episode thread and destroys
	/// the game. Returns the remaining stream and what was proven.
	EpisodeReport finish(bool natural, Deadline deadline);
	void closeEpisode(); ///< closes policy, joins existing native lifecycle, destroys game state
	SessionTimings timings() const;
	void waitForShutdown(Deadline deadline);
	bool shutdownComplete() const; ///< native headless success receipt, not independent replay verification
	bool poisoned() const noexcept;
	bool active() const noexcept;
};
}
VCMI_LIB_NAMESPACE_END
