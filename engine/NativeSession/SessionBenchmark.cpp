// Explicit native lifecycle workload: end-turn only, first displayed modal
// choice if required. This is an untrained test controller, never a learner.
#include "StdInc.h"
#include "EngineSession.h"
#include "ResourceUsage.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{
using Clock = std::chrono::steady_clock;
JsonNode parse(const std::string & text)
{
	JsonParsingSettings options;
	options.mode = JsonParsingSettings::JsonFormatMode::JSON;
	options.strict = true;
	return JsonNode(text.data(), text.size(), options, "native lifecycle plan");
}
void save(const std::filesystem::path & path, const JsonNode & value)
{
	std::ofstream stream(path);
	stream << value.toString() << '\n';
	stream.flush();
	if(!stream) throw std::runtime_error("Cannot persist native lifecycle receipt");
}
JsonNode response(const native_session::Boundary & boundary, const std::string & episode, int player)
{
	JsonNode decision;
	for(const auto & envelope : boundary.messages)
		if(envelope["records"].isVector())
			for(const auto & record : envelope["records"].Vector())
				if(record["kind"].String() == "decision") decision = record;
	if(decision.isNull()) throw std::runtime_error("No actual v7 decision at native boundary");
	JsonNode result;
	result["protocol_version"].Integer() = 9;
	result["type"].String() = "context_action";
	result["episode"].String() = episode;
	result["player"].Integer() = player;
	result["decision_id"] = decision["decision_id"];
	result["revision"] = decision["revision"];
	auto & action = result["action"];
	action["schema"].String() = "heroes-context-actions-v7";
	const auto & observation = decision["observation"];
	const auto & window = observation["window"];
	const auto kind = window.isNull() ? std::string("map") : window["context"]["kind"].String();
	if(kind == "map") action["kind"].String() = "end_turn";
	else if(kind == "dialog")
	{
		const auto & choices = observation["window"]["dialog_panel"]["choices"].Vector();
		if(choices.empty()) throw std::runtime_error("Missing displayed dialog choices");
		action["kind"].String() = "choose_dialog";
		action["subject"] = window["context"]["subject"];
		action["selection"] = choices.front()["ref"];
	}
	else throw std::runtime_error("End-turn workload reached an unsupported window; no fabricated action");
	return result;
}
JsonNode times(const native_session::SessionTimings & before, const native_session::SessionTimings & after)
{
	JsonNode value;
	value["load"].Float() = after.loadSeconds - before.loadSeconds;
	value["reset_through_first_boundary"].Float() = after.resetSeconds - before.resetSeconds;
	value["advance_inference_excluded"].Float() = after.advanceSeconds - before.advanceSeconds;
	value["cleanup"].Float() = after.cleanupSeconds - before.cleanupSeconds;
	return value;
}
}
int main(int argc, char ** argv)
{
	if(argc != 5 || std::string(argv[1]) != "--plan" || std::string(argv[3]) != "--output")
	{
		std::cerr << "Usage: native_session_benchmark --plan <plan.json> --output <new-directory>\n";
		return 2;
	}
	const std::filesystem::path output = std::filesystem::absolute(argv[4]);
	if(std::filesystem::exists(output)) return 2;
	std::filesystem::create_directories(output);
	JsonNode report;
	report["schema"].String() = "heroes-live-native-lifecycle-benchmark-v1";
	report["status"].String() = "incomplete";
	report["controller"].String() = "Explicit untrained end-turn-only workload; first displayed modal choice when required";
	report["replay_verified"].Bool() = false;
	report["episodes"].Vector();
	int code = 0;
	{
		native_session::EngineSession session;
		try
		{
			std::ifstream source(argv[2]);
			if(!source) throw std::runtime_error("Cannot read plan");
			const std::string contents((std::istreambuf_iterator<char>(source)), {});
			const auto plan = parse(contents);
			const auto & episodes = plan["episodes"].Vector();
			if(episodes.empty()) throw std::runtime_error("Plan requires explicit episodes");
			save(output / "plan.json", plan);
			session.load();
			report["load_seconds"].Float() = session.timings().loadSeconds;
			for(const auto & item : episodes)
			{
				native_session::EpisodeConfig config;
				config.episode = item["episode"].String();
				config.map = item["map"].String();
				config.seed = item["seed"].Integer();
				config.learner = item["learner"].Integer();
				const auto maxSteps = item["max_steps"].Integer();
				const auto seconds = item["seconds"].Integer();
				if(maxSteps < 1 || maxSteps > 1000 || seconds < 1 || seconds > 600)
					throw std::runtime_error("Bound each episode to 1..1000 steps and 1..600 seconds");
				const auto folder = output / ("episode-" + std::to_string(report["episodes"].Vector().size()));
				std::filesystem::create_directory(folder);
				config.captureDirectory = (folder / "engine").string();
				JsonNode receipt;
				receipt["config"] = item;
				receipt["status"].String() = "incomplete";
				receipt["capture_directory"].String() = config.captureDirectory;
				report["episodes"].Vector().push_back(receipt);
				auto & current = report["episodes"].Vector().back();
				save(output / "summary.json", report);
				const auto before = session.timings();
				const auto started = Clock::now();
				const auto deadline = started + std::chrono::seconds(seconds);
				session.reset(config);
				auto boundary = session.observe(deadline);
				std::ofstream bridge(folder / "player-messages.jsonl");
				int64_t steps = 0;
				auto persist = [&]
				{
					for(const auto & message : boundary.messages)
					{
						auto text = message.toCompactString();
						text.erase(std::remove(text.begin(), text.end(), '\n'), text.end());
						bridge << text << '\n';
					}
					bridge.flush();
				};
				persist();
				while(!boundary.terminalObserved && steps < maxSteps)
				{
					if(!boundary.needsAction) throw std::runtime_error("Native boundary has no decision");
					const auto action = response(boundary, config.episode, config.learner);
					const auto stepStart = Clock::now();
					boundary = session.step(boundary.token, action, deadline);
					current["step_seconds"].Vector().emplace_back(std::chrono::duration<double>(Clock::now() - stepStart).count());
					++steps;
					persist();
				}
				if(boundary.terminalObserved) session.waitForShutdown(deadline);
				current["native_shutdown_complete"].Bool() = session.shutdownComplete();
				current["status"].String() = boundary.terminalObserved ? "natural_unverified" : "prefix_unverified";
				current["submitted_steps"].Integer() = steps;
				session.closeEpisode();
				current["timing_seconds"] = times(before, session.timings());
				current["wall_seconds"].Float() = std::chrono::duration<double>(Clock::now() - started).count();
				if(const auto peak = native_session_host::peakResidentBytes()) current["peak_process_resident_bytes"].Integer() = *peak;
				save(output / "summary.json", report);
			}
			report["status"].String() = "completed_pending_independent_replay";
		}
		catch(const std::exception & error)
		{
			report["status"].String() = "failed";
			report["error"].String() = error.what();
			code = 3;
			session.closeEpisode();
		}
	}
	if(const auto peak = native_session_host::peakResidentBytes()) report["peak_process_resident_bytes"].Integer() = *peak;
	save(output / "summary.json", report);
	return code;
}
