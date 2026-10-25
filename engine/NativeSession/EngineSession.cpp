#include "StdInc.h"
#include "EngineSession.h"
#include "RuntimeEnvironment.h"
#include "SessionHealth.h"
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
#include "../lib/logging/CLogger.h"
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>

namespace
{
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::mutex ownerMutex;
std::atomic<uint64_t> nextGeneration{1};
std::atomic<bool> resourcesEverLoaded{false};
struct Shutdown : native_session_host::ShutdownStatus
{
	std::mutex mutex;
	std::condition_variable changed;
	std::weak_ptr<native_session::InProcessPolicy> policy;
};
std::mutex activeMutex;
std::shared_ptr<Shutdown> activeShutdown;
std::shared_ptr<Shutdown> currentShutdown()
{
	std::lock_guard lock(activeMutex);
	return activeShutdown;
}

/// One game's engine log, in the console format and threshold the executable
/// client writes to its redirected stdout (logs/process.log).
struct EpisodeLogState
{
	std::mutex mutex;
	std::FILE * file = nullptr;
	bool failed = false;
	CLogFormatter formatter;
	ELogLevel::ELogLevel threshold = ELogLevel::INFO;
};
class EpisodeLogTarget final : public ILogTarget
{
	std::shared_ptr<EpisodeLogState> state;
public:
	explicit EpisodeLogTarget(std::shared_ptr<EpisodeLogState> value) : state(std::move(value)) {}
	void write(const LogRecord & record) override
	{
		std::lock_guard lock(state->mutex);
		if(!state->file || state->threshold > record.level)
			return;
		const auto text = state->formatter.format(record) + "\n";
		if(std::fwrite(text.data(), 1, text.size(), state->file) != text.size())
			state->failed = true;
	}
};
ELogLevel::ELogLevel logLevel(const std::string & name)
{
	static const std::map<std::string, ELogLevel::ELogLevel> levels{{"trace", ELogLevel::TRACE},
		{"debug", ELogLevel::DEBUG}, {"info", ELogLevel::INFO}, {"warn", ELogLevel::WARN}, {"error", ELogLevel::ERROR}};
	const auto found = levels.find(name);
	if(found == levels.end()) throw std::runtime_error("Unknown console log threshold " + name);
	return found->second;
}
void restore(SettingsStorage & storage, const JsonNode & baseline)
{
	Settings root = storage.write(std::vector<std::string>{});
	*root.operator->() = baseline;
}
}

// Executable host hooks, deliberately outside the public gameplay API.
void requestHeadlessShutdown(int code)
{
	if(auto value = currentShutdown())
	{
		std::lock_guard lock(value->mutex);
		if(!value->record(code)) return;
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
	const SessionOptions options;
	CConsoleHandler console;
	std::unique_ptr<CBasicLogConfigurator> logging;
	std::shared_ptr<EpisodeLogState> episodeLog = std::make_shared<EpisodeLogState>();
	std::unique_ptr<native_session_host::RuntimeEnvironment> environment;
	std::shared_ptr<InProcessPolicy> policy;
	std::shared_ptr<Shutdown> shutdown;
	native_session_host::NullkillerOpponent opponent;
	native_session_host::OpponentStats lastOpponent;
	JsonNode settingsBaseline;
	JsonNode persistentBaseline;
	SessionTimings timing;
	bool advancing = false; ///< a response was applied; its next boundary is not yet delivered
	int player = -1;
	bool loaded = false;
	bool awaitingInitialBoundary = false;
	bool opponentAttached = false;
	native_session_host::SessionHealth health;
	explicit Impl(SessionOptions value) : options(value)
	{
		if(!owner.owns_lock() || LIBRARY || GAME || ENGINE)
			throw std::runtime_error("Native EngineSession requires exclusive ownership of a fresh process");
	}
	void openEpisodeLog(const std::string & path)
	{
		if(path.empty()) return;
		std::lock_guard lock(episodeLog->mutex);
		if(episodeLog->file) throw std::logic_error("Previous native game log is still open");
		episodeLog->file = std::fopen(path.c_str(), "wx");
		episodeLog->failed = false;
		if(!episodeLog->file) throw std::runtime_error("Cannot create the new native game log " + path);
	}
	void closeEpisodeLog()
	{
		std::lock_guard lock(episodeLog->mutex);
		if(!episodeLog->file) return;
		const bool flushed = std::fflush(episodeLog->file) == 0;
		const bool closed = std::fclose(episodeLog->file) == 0;
		episodeLog->file = nullptr;
		if(!flushed || !closed || episodeLog->failed)
			throw std::runtime_error("Native game log is incomplete");
	}
};
EngineSession::EngineSession(SessionOptions options) : impl(std::make_unique<Impl>(options)) {}
EngineSession::~EngineSession()
{
	try { closeEpisode(); }
	catch(...)
	{
		// Never destroy resources beneath live episode threads. A failure after
		// the game was destroyed (its log or the opponent module reset) is not one.
		if(GAME || impl->policy) std::terminate();
	}
	impl->opponent.shutdown();
	setExperimentServerFailureCapture(false);
	setExperimentFailureHandler({});
	setExperimentEvidenceHooks(nullptr);
	delete LIBRARY;
	LIBRARY = nullptr;
	if(impl->logging) impl->logging->deconfigure();
}

void EngineSession::load()
{
	if(impl->health.failed() || impl->loaded || resourcesEverLoaded.exchange(true))
		throw std::logic_error("Native load requires a fresh process; reuse reset() on the original loaded owner");
	const auto start = Clock::now();
	try
	{
		const auto sessionLog = VCMIDirs::get().userLogsPath() / "VCMI_Session_log.txt";
		impl->logging = std::make_unique<CBasicLogConfigurator>(sessionLog, &impl->console);
		impl->logging->configureDefault();
		LIBRARY = new GameLibrary;
		LIBRARY->initializeFilesystem(false);
		impl->logging->configure();
		const auto & console = settings["logging"]["console"];
		impl->episodeLog->formatter = CLogFormatter(console["format"].String());
		impl->episodeLog->threshold = logLevel(console["threshold"].String());
		if(!impl->options.consoleLog)
		{
			// An embedding host owns its stdout. Keep the persistent session file
			// and the per-game log; never interleave engine lines with the host.
			auto * global = CLogger::getGlobalLogger();
			global->clearTargets();
			auto file = std::make_unique<CLogFileTarget>(sessionLog, true);
			if(!settings["logging"]["file"]["format"].isNull())
				file->setFormatter(CLogFormatter(settings["logging"]["file"]["format"].String()));
			global->addTarget(std::move(file));
		}
		CLogger::getGlobalLogger()->addTarget(std::make_unique<EpisodeLogTarget>(impl->episodeLog));
		LIBRARY->initializeLibrary();
		impl->opponent.initialize();
		// A failure on the in-thread server poisons this game, never the host process.
		setExperimentServerFailureCapture(true);
		// Each game starts from the loaded configuration, as each executable
		// launch starts from its freshly written profile.
		impl->settingsBaseline = settings.toJsonNode();
		impl->persistentBaseline = persistentStorage.toJsonNode();
		impl->loaded = true;
	}
	catch(...) { impl->timing.loadSeconds = elapsed(start); impl->health.poison(); throw; }
	impl->timing.loadSeconds = elapsed(start);
}

void EngineSession::reset(const EpisodeConfig & config)
{
	impl->health.requireResetReady(impl->loaded);
	if(config.episode.empty() || config.map.empty() || config.seed <= 0 || config.learner < 0 || config.learner > 1
		|| config.policyTimeoutSeconds < 1 || config.policyTimeoutSeconds > 300 || config.interface.empty()
		|| config.interface.size() > 64
		|| config.interface.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos)
		throw std::invalid_argument("Invalid native episode identity/configuration");
	const std::filesystem::path capture(config.captureDirectory);
	if(!capture.is_absolute() || std::filesystem::exists(capture))
		throw std::invalid_argument("Native capture directory must be new and absolute");
	if(!config.logPath.empty())
	{
		const std::filesystem::path log(config.logPath);
		if(!log.is_absolute() || std::filesystem::exists(log) || !std::filesystem::is_directory(log.parent_path()))
			throw std::invalid_argument("Native game log must be a new absolute file in an existing directory");
	}
	impl->health.prepareReset(impl->loaded, [&] { closeEpisode(); });
	const auto start = Clock::now();
	try
	{
		std::filesystem::create_directories(capture);
		impl->openEpisodeLog(config.logPath);
		restore(settings, impl->settingsBaseline);
		restore(persistentStorage, impl->persistentBaseline);
		impl->environment = std::make_unique<native_session_host::RuntimeEnvironment>();
		auto & env = *impl->environment;
		env.set("EXPLORER_RL_PLAYER", std::nullopt);
		env.set("EXPLORER_RL_PLAYERS", std::to_string(config.learner));
		env.set("EXPLORER_RL_BUILTIN_OPPONENT", native_session_host::NullkillerOpponent::MODULE);
		env.set("EXPLORER_RL_INTERFACE", config.interface);
		env.set("EXPLORER_RL_TRANSPORT", "in-process");
		env.set("EXPLORER_RL_EPISODE", config.episode);
		env.set("EXPLORER_RL_TIMEOUT", std::to_string(config.policyTimeoutSeconds));
		env.set("EXPLORER_RL_PROFILE", "0");
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
		resetNativeSessionEvidence(); // previous native objects and threads are gone
		// Room for a whole opponent turn of visible events; take() wakes at half.
		impl->policy = std::make_shared<InProcessPolicy>(nextGeneration.fetch_add(1), 1024);
		impl->player = config.learner;
		installPolicy(config.learner, config.episode, impl->policy);
		impl->shutdown = std::make_shared<Shutdown>();
		impl->shutdown->policy = impl->policy;
		{
			std::lock_guard lock(activeMutex);
			activeShutdown = impl->shutdown;
		}
		// The executable headless client's handler: the reason reaches the game log.
		setExperimentFailureHandler([](const std::string & reason)
		{
			logGlobal->error("Experiment failure: %s", reason);
			requestHeadlessShutdown(EXIT_FAILURE);
		});
		impl->lastOpponent = {};
		impl->opponent.attach(1 - config.learner);
		impl->opponentAttached = true;
		impl->advancing = false;
		GAME = std::make_unique<GameInstance>();
		GAME->server().debugStartTest(config.map, false);
		impl->awaitingInitialBoundary = true;
		++impl->timing.episodes;
	}
	catch(...) { impl->health.poison(); closeEpisode(); throw; }
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
	// A response superseded by the player's own terminal was never applied: no step, no advance time.
	const bool applied = impl->policy->submit(token, response);
	if(applied) ++impl->timing.steps;
	try
	{
		auto result = observe(deadline);
		if(applied) impl->timing.advanceSeconds += elapsed(start);
		return result;
	}
	catch(...)
	{
		if(applied) impl->timing.advanceSeconds += elapsed(start);
		throw;
	}
}
Delivery EngineSession::take(Deadline deadline, bool untilBoundary)
{
	if(!impl->policy) throw std::logic_error("No active native episode");
	const auto start = Clock::now();
	auto result = impl->policy->take(deadline, untilBoundary);
	const bool boundary = result.needsAction || result.terminalObserved || result.closed || !result.failure.empty();
	if(impl->awaitingInitialBoundary)
	{
		impl->timing.resetSeconds += elapsed(start);
		if(boundary) impl->awaitingInitialBoundary = false;
	}
	else if(impl->advancing)
	{
		// Only time spent waiting here: host-side work between takes is not engine time.
		impl->timing.advanceSeconds += elapsed(start);
		if(boundary) impl->advancing = false;
	}
	return result;
}
bool EngineSession::submit(BoundaryToken token, const JsonNode & response, bool lineAdmitted)
{
	if(!impl->policy) throw std::logic_error("No active native episode");
	if(!impl->policy->submit(token, response, lineAdmitted)) return false;
	++impl->timing.steps;
	impl->advancing = true;
	return true;
}
EpisodeReport EngineSession::finish(bool natural, Deadline deadline)
{
	if(!impl->policy || !impl->shutdown) throw std::logic_error("No active native episode");
	EpisodeReport report;
	report.natural = natural;
	const auto policy = impl->policy;
	const auto shutdown = impl->shutdown;
	const auto start = Clock::now();
	std::optional<int> before;
	{
		std::unique_lock lock(shutdown->mutex);
		if(natural && !shutdown->changed.wait_until(lock, deadline, [&] { return shutdown->code.has_value(); }))
			report.shutdownDeadline = true;
		before = shutdown->code;
		// From this snapshot on, a success status can no longer appear; a failure still can.
		shutdown->stopping = true;
	}
	// The natural terminal barrier never completed: a native shutdown failure.
	if(report.shutdownDeadline) impl->health.poison();
	report.waitSeconds = elapsed(start);
	if(natural) impl->timing.finishWaitSeconds += report.waitSeconds;
	report.hostStopped = !before.has_value();
	const auto closing = Clock::now();
	try { closeEpisode(); }
	catch(const std::exception & error) { report.closeError = error.what(); }
	catch(...) { report.closeError = "Native episode close failed"; }
	report.closeSeconds = elapsed(closing);
	{
		std::lock_guard lock(shutdown->mutex);
		report.exitCode = shutdown->code; // a failure reported while stopping survives
	}
	report.channelFailure = policy->failure();
	report.messages = policy->drain(); // every producer has joined
	report.experimentFailed = experimentHasFailed();
	report.opponent = impl->lastOpponent;
	report.poisoned = impl->health.failed();
	report.closeClean = report.closeError.empty() && !report.poisoned && !report.experimentFailed
		&& report.channelFailure.empty() && !(report.exitCode && *report.exitCode != 0)
		&& report.opponent.released && (!report.opponent.created || report.opponent.moduleReset) && !report.shutdownDeadline;
	return report;
}
void EngineSession::closeEpisode()
{
	if(!impl->policy && !GAME && !impl->environment && !impl->opponentAttached) return;
	const auto start = Clock::now();
	try
	{
		if(impl->shutdown)
		{
			std::lock_guard lock(impl->shutdown->mutex);
			impl->shutdown->stopping = true;
			if(impl->shutdown->failed()) impl->health.poison();
		}
		if(impl->policy) impl->policy->close();
		if(GAME)
		{
			logGlobal->info("Native session cleanup: stop/join client network");
			GAME->server().endNetwork();
			logGlobal->info("Native session cleanup: finish AI and game state");
			if(GAME->server().client) GAME->server().endGameplay();
			logGlobal->info("Native session cleanup: destroy server runner");
			// CServerHandler destruction shuts down and joins its server runner.
			// Keep GAME non-null throughout destruction: endGame uses the global.
			auto * game = GAME.get();
			delete game;
			GAME.release();
		}
		if(impl->opponentAttached)
		{
			// Every thread and object of the game is gone. An opponent instance
			// that survived it would carry state into the next game.
			impl->opponentAttached = false;
			impl->lastOpponent = impl->opponent.release(); // never throws
			impl->timing.opponentTurns += impl->lastOpponent.turns;
			impl->timing.opponentTurnSeconds += impl->lastOpponent.turnSeconds;
			impl->timing.battles += impl->lastOpponent.battles;
			impl->timing.battleSeconds += impl->lastOpponent.battleSeconds;
			if(!impl->lastOpponent.released)
			{
				logGlobal->error("Native session cleanup: built-in opponent instance outlived its game");
				impl->health.poison();
			}
			else if(impl->lastOpponent.created && !impl->lastOpponent.moduleReset)
			{
				// The next game would start with this game's Nullkiller module caches.
				logGlobal->error("Native session cleanup: built-in opponent module state was not reset");
				impl->health.poison();
			}
		}
		// All producers have joined. Preserve errors published before cancellation
		// or during shutdown before discarding the channel and resetting evidence.
		if(experimentHasFailed() || (impl->policy && impl->policy->failed()))
			impl->health.poison();
		if(impl->shutdown)
		{
			std::lock_guard lock(impl->shutdown->mutex);
			if(impl->shutdown->failed()) impl->health.poison();
		}
		if(impl->player >= 0) removePolicy(impl->player);
		impl->player = -1;
		impl->policy.reset();
		setExperimentFailureHandler({});
		auto evidence = experimentEvidenceHooks();
		setExperimentEvidenceHooks(nullptr);
		logGlobal->info("Native session cleanup: release contextual evidence");
		if(evidence) evidence->finishNativeSession();
		{
			std::lock_guard lock(activeMutex);
			activeShutdown.reset();
		}
		impl->environment.reset();
		impl->closeEpisodeLog();
		impl->timing.cleanupSeconds += elapsed(start);
	}
	catch(...)
	{
		impl->health.poison();
		try { impl->closeEpisodeLog(); } catch(...) {}
		throw;
	}
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
bool EngineSession::poisoned() const noexcept { return impl->health.failed(); }
bool EngineSession::active() const noexcept { return impl->policy != nullptr; }
}
VCMI_LIB_NAMESPACE_END
