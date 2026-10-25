/* Explicitly owned built-in opponent of an embedded game. GPL-2.0-or-later. */
#pragma once
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

VCMI_LIB_NAMESPACE_BEGIN
class CGlobalAI;
struct CPackForClient;
VCMI_LIB_NAMESPACE_END

namespace native_session_host
{
/// Counters are observed from applied client packets; they never steer play.
struct OpponentStats
{
	std::string module;
	int player = -1;
	int difficulty = -1; ///< requested StartInfo difficulty; the engine metadata proves the effective one
	bool created = false; ///< the engine installed exactly this owned instance
	bool released = false; ///< the instance was destroyed together with its game
	bool moduleReset = false; ///< module caches were returned to fresh-process state
	uint64_t turns = 0; ///< completed adventure turns of the opponent seat
	double turnSeconds = 0; ///< PlayerStartsTurn..PlayerEndsTurn wall time, engine work included
	uint64_t battles = 0; ///< automatic battles of either side (BattleAI), started and ended
	double battleSeconds = 0; ///< BattleStart..BattleEnded wall time, engine work included
};

/// Unchanged Nullkiller on Easy for the seat the learner does not control.
/// Lifecycle: initialize() once after resources load; attach() before each
/// game; the engine's own turn order advances it (the host never drives or
/// inspects it); detach() and release() after the game's threads are joined;
/// shutdown() with the session. A new AIGateway is created for every game.
class NullkillerOpponent
{
public:
	static constexpr const char * MODULE = "Nullkiller";
	static constexpr int DIFFICULTY_EASY = 0;
	void initialize();
	/// Installs the opponent provider and passive packet observer, and the
	/// unchanged Easy/BattleAI selections. No game client may exist yet.
	void attach(int player);
	void detach() noexcept;
	/// Requires detach() and a destroyed game. Not released means poisoned.
	OpponentStats release();
	void shutdown() noexcept;
	bool initialized() const noexcept { return ready; }
private:
	using Clock = std::chrono::steady_clock;
	std::shared_ptr<VCMI_LIB_WRAP_NAMESPACE(CGlobalAI)> create(const std::string & module, int player);
	void observe(const VCMI_LIB_WRAP_NAMESPACE(CPackForClient) & pack);
	mutable std::mutex mutex;
	OpponentStats stats;
	std::weak_ptr<VCMI_LIB_WRAP_NAMESPACE(CGlobalAI)> instance;
	std::optional<Clock::time_point> turnStart;
	std::map<int, Clock::time_point> battleStarts;
	bool ready = false;
	bool attached = false;
};
}
