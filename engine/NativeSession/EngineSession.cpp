#include "StdInc.h"
#include "EngineSession.h"
#include "RuntimeEnvironment.h"
#include "../client/CMT.h"
#include "../client/Client.h"
#include "../client/CServerHandler.h"
#include "../client/GameEngine.h"
#include "../client/GameInstance.h"
#include "../lib/CConfigHandler.h"
#include "../lib/CConsoleHandler.h"
#include "../lib/GameLibrary.h"
#include "../lib/VCMIDirs.h"
#include "../lib/callback/CGameInterface.h"
#include "../lib/callback/ExperimentHooks.h"
#include "../lib/logging/CBasicLogConfigurator.h"
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <optional>
#include <set>

namespace
{
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::mutex ownerMutex;
std::atomic<uint64_t> nextGeneration{1};
std::atomic<bool> resourcesEverLoaded{false};
struct Shutdown
{
	std::mutex mutex;
	std::condition_variable changed;
	std::optional<int> code;
	bool stopping = false;
	std::weak_ptr<native_session::InProcessPolicy> policy;
};
std::mutex activeMutex;
std::shared_ptr<Shutdown> active;
std::shared_ptr<Shutdown> currentShutdown()
{
	std::lock_guard lock(activeMutex);
	return active;
}
}

// Executable host hooks, deliberately outside the public gameplay API.
void requestHeadlessShutdown(int code)
{
	if(auto value = currentShutdown())
	{
		std::lock_guard lock(value->mutex);
		if(value->stopping) return; // intentional cleanup is not a new game outcome
		if(!value->code || code != 0) value->code = code;
		value->changed.notify_all();
		if(code != 0 && !value->stopping)
			if(auto policy = value->policy.lock()) policy->fail("Native headless shutdown failed");
	}
}
bool headlessShutdownRequested()
{
	if(auto value = currentShutdown())
	{
		std::lock_guard lock(value->mutex);
		return value->stopping || value->code.has_value();
	}
	return false;
}
[[noreturn]] void handleFatalError(const std::string & message, bool)
{
	reportExperimentFailure(message);
	throw std::runtime_error(message);
}

VCMI_LIB_NAMESPACE_BEGIN
namespace native_session
{
class EngineSession::Impl
{
public:
	std::unique_lock<std::mutex> owner{ownerMutex, std::try_to_lock};
	CConsoleHandler console;
	std::unique_ptr<CBasicLogConfigurator> logging;
	std::unique_ptr<native_session_host::RuntimeEnvironment> environment;
	std::shared_ptr<InProcessPolicy> policy;
	std::shared_ptr<Shutdown> shutdown;
	std::set<std::string> episodes;
	SessionTimings timing;
	int player = -1;
	bool loaded = false;
	bool awaitingInitialBoundary = false;
	bool poisoned = false;
	Impl()
	{
		if(!owner.owns_lock() || LIBRARY || GAME || ENGINE)
			throw std::runtime_error("Native EngineSession requires exclusive ownership of a fresh process");
	}
};
EngineSession::EngineSession() : impl(std::make_unique<Impl>()) {}
EngineSession::~EngineSession()
{
	try { closeEpisode(); } catch(...) { std::terminate(); } // never destroy resources beneath live episode threads
	setExperimentFailureHandler({});
	setExperimentEvidenceHooks(nullptr);
	delete LIBRARY;
	LIBRARY = nullptr;
	if(impl->logging) impl->logging->deconfigure();
}

void EngineSession::load()
{
	if(impl->poisoned || impl->loaded || resourcesEverLoaded.exchange(true))
		throw std::logic_error("Native load requires a fresh process; reuse reset() on the original loaded owner");
	const auto start = Clock::now();
	try
	{
		impl->logging = std::make_unique<CBasicLogConfigurator>(VCMIDirs::get().userLogsPath() / "VCMI_Session_log.txt", &impl->console);
		impl->logging->configureDefault();
		LIBRARY = new GameLibrary;
		LIBRARY->initializeFilesystem(false);
		impl->logging->configure();
		LIBRARY->initializeLibrary();
		impl->loaded = true;
	}
	catch(...) { impl->timing.loadSeconds = elapsed(start); impl->poisoned = true; throw; }
	impl->timing.loadSeconds = elapsed(start);
}

void EngineSession::reset(const EpisodeConfig & config)
{
	if(!impl->loaded || impl->poisoned) throw std::logic_error("Load resources first; failed native sessions require a fresh process");
	if(config.episode.empty() || config.map.empty() || config.seed <= 0 || config.learner < 0 || config.learner > 1
		|| config.policyTimeoutSeconds < 1 || config.policyTimeoutSeconds > 300 || impl->episodes.contains(config.episode))
		throw std::invalid_argument("Invalid or reused native episode identity/configuration");
	const std::filesystem::path capture(config.captureDirectory);
	if(!capture.is_absolute() || std::filesystem::exists(capture))
		throw std::invalid_argument("Native capture directory must be new and absolute");
	closeEpisode();
	const auto start = Clock::now();
	try
	{
		std::filesystem::create_directories(capture);
		impl->episodes.insert(config.episode);
		impl->environment = std::make_unique<native_session_host::RuntimeEnvironment>();
		auto & env = *impl->environment;
		env.set("EXPLORER_RL_PLAYER", std::nullopt);
		env.set("EXPLORER_RL_PLAYERS", std::to_string(config.learner));
		env.set("EXPLORER_RL_BUILTIN_OPPONENT", "Nullkiller");
		env.set("EXPLORER_RL_INTERFACE", "adventure-dialogs-v1");
		env.set("EXPLORER_RL_TRANSPORT", "in-process");
		env.set("EXPLORER_RL_EPISODE", config.episode);
		env.set("EXPLORER_RL_TIMEOUT", std::to_string(config.policyTimeoutSeconds));
		env.set("VCMI_EXPERIMENT_DIR", capture.string());
		env.set("VCMI_RECORD_DIR", std::nullopt);
		env.set("VCMI_EXPERIMENT_CONTEXTUAL_EVIDENCE", "1");
		env.set("VCMI_EXPERIMENT_REPLAY", "1");
		env.set("VCMI_EXPERIMENT_NO_CHEATS", "1");
		// Host-controlled transport/lifecycle only. All game actions still pass
		// through the same ExplorerRL/CCallback/server request/packet path.
		{
			Settings session = settings.write["session"];
			session["headless"].Bool() = true;
			session["onlyai"].Bool() = true;
			session["spectate"].Bool() = false;
			session["donotstartserver"].Bool() = false;
			session["testmap"].String() = config.map;
		}
		{
			Settings server = settings.write["server"];
			server["useProcess"].Bool() = false;
			server["seed"].Integer() = config.seed;
		}
		{
			Settings general = settings.write["general"];
			general["lastDifficulty"].Integer() = 0;
		}
		{
			Settings ai = settings.write["ai"];
			ai["combatEnemyAI"].String() = "BattleAI";
			ai["combatAlliedAI"].String() = "BattleAI";
			ai["combatNeutralAI"].String() = "BattleAI";
		}
		resetNativeSessionEvidence(); // previous native objects and threads are gone
		impl->policy = std::make_shared<InProcessPolicy>(nextGeneration.fetch_add(1));
		impl->player = config.learner;
		installPolicy(config.learner, config.episode, impl->policy);
		impl->shutdown = std::make_shared<Shutdown>();
		impl->shutdown->policy = impl->policy;
		{
			std::lock_guard lock(activeMutex);
			active = impl->shutdown;
		}
		setExperimentFailureHandler([](const std::string &) { requestHeadlessShutdown(EXIT_FAILURE); });
		GAME = std::make_unique<GameInstance>();
		GAME->server().debugStartTest(config.map, false);
		impl->awaitingInitialBoundary = true;
		++impl->timing.episodes;
	}
	catch(...) { impl->poisoned = true; closeEpisode(); throw; }
	impl->timing.resetSeconds += elapsed(start);
}

Boundary EngineSession::observe(Deadline deadline)
{
	if(!impl->policy) throw std::logic_error("No active native episode");
	const auto start = Clock::now();
	try
	{
		auto result = impl->policy->observe(deadline);
		if(impl->awaitingInitialBoundary)
		{
			impl->timing.resetSeconds += elapsed(start);
			impl->awaitingInitialBoundary = false;
		}
		return result;
	}
	catch(...)
	{
		if(impl->awaitingInitialBoundary) impl->timing.resetSeconds += elapsed(start);
		throw;
	}
}
Boundary EngineSession::step(BoundaryToken token, const JsonNode & response, Deadline deadline)
{
	if(!impl->policy) throw std::logic_error("No active native episode");
	const auto start = Clock::now();
	impl->policy->submit(token, response);
	++impl->timing.steps;
	try
	{
		auto result = observe(deadline);
		impl->timing.advanceSeconds += elapsed(start);
		return result;
	}
	catch(...) { impl->timing.advanceSeconds += elapsed(start); throw; }
}
void EngineSession::closeEpisode()
{
	if(!impl->policy && !GAME && !impl->environment) return;
	const auto start = Clock::now();
	if(impl->shutdown)
	{
		std::lock_guard lock(impl->shutdown->mutex);
		impl->shutdown->stopping = true;
		if(impl->shutdown->code && *impl->shutdown->code != 0) impl->poisoned = true;
	}
	if(impl->policy) impl->policy->close();
	if(GAME)
	{
		GAME->server().endNetwork();
		if(GAME->server().client) GAME->server().endGameplay();
		// CServerHandler destruction shuts down and joins its server runner.
		// Keep GAME non-null throughout destruction: endGame uses the global.
		auto * game = GAME.get();
		delete game;
		GAME.release();
	}
	if(impl->player >= 0) removePolicy(impl->player);
	impl->player = -1;
	impl->policy.reset();
	setExperimentFailureHandler({});
	auto evidence = experimentEvidenceHooks();
	setExperimentEvidenceHooks(nullptr);
	if(evidence) evidence->finishNativeSession();
	{
		std::lock_guard lock(activeMutex);
		active.reset();
	}
	impl->environment.reset();
	impl->timing.cleanupSeconds += elapsed(start);
}
SessionTimings EngineSession::timings() const { return impl->timing; }
void EngineSession::waitForShutdown(Deadline deadline)
{
	if(!impl->shutdown) throw std::logic_error("No native episode shutdown state");
	std::unique_lock lock(impl->shutdown->mutex);
	if(!impl->shutdown->changed.wait_until(lock, deadline, [&] { return impl->shutdown->code.has_value(); }))
		throw std::runtime_error("Native terminal barrier/shutdown deadline");
	if(*impl->shutdown->code != 0 || experimentHasFailed())
		throw std::runtime_error("Native terminal barrier/shutdown failed");
}
bool EngineSession::shutdownComplete() const
{
	if(!impl->shutdown) return false;
	std::lock_guard lock(impl->shutdown->mutex);
	return impl->shutdown->code == 0;
}
}
VCMI_LIB_NAMESPACE_END
