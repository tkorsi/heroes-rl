/* Experimental live VCMI session. GPL-2.0-or-later. */
#pragma once
#include "PolicyTransport.h"
#include <memory>
#include <string>

VCMI_LIB_NAMESPACE_BEGIN
namespace native_session
{
struct EpisodeConfig
{
	std::string episode;
	std::string map; ///< existing VCMI resource URI, as used by --testmap
	std::string captureDirectory; ///< new, absolute directory; never reused
	int seed = 0;
	int learner = 0; ///< other side is the unchanged Nullkiller Easy
	int policyTimeoutSeconds = 30;
};
struct SessionTimings
{
	double loadSeconds = 0;
	double resetSeconds = 0; ///< startup dispatch plus waits through first held boundary; caller idle excluded
	double advanceSeconds = 0; ///< submit-to-next-boundary, excludes caller inference
	double cleanupSeconds = 0;
	uint64_t episodes = 0;
	uint64_t steps = 0;
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
	EngineSession();
	~EngineSession();
	EngineSession(const EngineSession &) = delete;
	EngineSession & operator=(const EngineSession &) = delete;
	void load();
	void reset(const EpisodeConfig & config);
	Boundary observe(Deadline deadline);
	Boundary step(BoundaryToken token, const JsonNode & response, Deadline deadline);
	void closeEpisode(); ///< closes policy, joins existing native lifecycle, destroys game state
	SessionTimings timings() const;
	void waitForShutdown(Deadline deadline);
	bool shutdownComplete() const; ///< native headless success receipt, not independent replay verification
};
}
VCMI_LIB_NAMESPACE_END
