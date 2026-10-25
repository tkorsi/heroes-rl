#include "StdInc.h"
#include "Opponent.h"
#include "../lib/CConfigHandler.h"
#include "../lib/VCMIDirs.h"
#include "../lib/callback/CDynLibHandler.h"
#include "../lib/callback/CGlobalAI.h"
#include "../lib/callback/ExperimentHooks.h"
#include "../lib/networkPacks/PacksForClient.h"
#include "../lib/networkPacks/PacksForClientBattle.h"
#include <stdexcept>

namespace native_session_host
{
void NullkillerOpponent::initialize()
{
	std::lock_guard lock(mutex);
	if(ready) return;
	const auto library = VCMIDirs::get().fullLibraryPath("AI", MODULE);
	if(!boost::filesystem::is_regular_file(library))
		throw std::runtime_error("Built-in opponent module is missing: " + library.string());
	ready = true;
}

void NullkillerOpponent::attach(int player)
{
	{
		std::lock_guard lock(mutex);
		if(!ready || attached || player < 0 || player > 1)
			throw std::logic_error("Attach an initialized, released opponent to Red or Blue");
		stats = OpponentStats{};
		stats.module = MODULE;
		stats.player = player;
		stats.difficulty = DIFFICULTY_EASY;
		instance.reset();
		turnStart.reset();
		battleStarts.clear();
		attached = true;
	}
	// The same selections the executable launcher writes: Easy, and BattleAI
	// for every automatic battle side. They are read when the game starts.
	{
		Settings general = settings.write["general"];
		general["lastDifficulty"].Integer() = DIFFICULTY_EASY;
	}
	{
		Settings ai = settings.write["ai"];
		ai["combatEnemyAI"].String() = "BattleAI";
		ai["combatAlliedAI"].String() = "BattleAI";
		ai["combatNeutralAI"].String() = "BattleAI";
	}
	CDynLibHandler::setExperimentOpponentProvider([this](const std::string & module, int seat) { return create(module, seat); });
	setExperimentPacketObserver([this](const CPackForClient & pack) { observe(pack); });
}

std::shared_ptr<CGlobalAI> NullkillerOpponent::create(const std::string & module, int player)
{
	std::lock_guard lock(mutex);
	if(!attached || module != MODULE || player != stats.player || stats.created)
		throw std::runtime_error("Engine requested an unexpected built-in opponent");
	// The engine's ordinary loader: a new, unmodified AIGateway per game.
	auto result = CDynLibHandler::getNewAI(module);
	instance = result;
	stats.created = true;
	return result;
}

void NullkillerOpponent::observe(const CPackForClient & pack)
{
	const auto now = Clock::now();
	std::lock_guard lock(mutex);
	if(const auto * started = dynamic_cast<const PlayerStartsTurn *>(&pack))
	{
		if(started->player.getNum() == stats.player)
			turnStart = now;
	}
	else if(const auto * ended = dynamic_cast<const PlayerEndsTurn *>(&pack))
	{
		if(ended->player.getNum() == stats.player && turnStart)
		{
			++stats.turns;
			stats.turnSeconds += std::chrono::duration<double>(now - *turnStart).count();
			turnStart.reset();
		}
	}
	else if(const auto * battle = dynamic_cast<const BattleStart *>(&pack))
		battleStarts[battle->battleID.getNum()] = now;
	else if(const auto * finished = dynamic_cast<const BattleEnded *>(&pack))
	{
		const auto found = battleStarts.find(finished->battleID.getNum());
		if(found != battleStarts.end())
		{
			++stats.battles;
			stats.battleSeconds += std::chrono::duration<double>(now - found->second).count();
			battleStarts.erase(found);
		}
	}
}

void NullkillerOpponent::detach() noexcept
{
	CDynLibHandler::setExperimentOpponentProvider({});
	setExperimentPacketObserver({});
}

OpponentStats NullkillerOpponent::release()
{
	detach();
	std::lock_guard lock(mutex);
	if(!attached)
		return stats;
	attached = false;
	stats.released = instance.expired();
	// Never touch module state beneath a surviving instance or its tasks.
	if(stats.released && stats.created)
		stats.moduleReset = CDynLibHandler::resetExperimentModule(MODULE);
	return stats;
}

void NullkillerOpponent::shutdown() noexcept
{
	detach();
	std::lock_guard lock(mutex);
	ready = false;
}
}
