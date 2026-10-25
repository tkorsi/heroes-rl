#include "StdInc.h"
#include "EngineSession.h"
#include "ResourceUsage.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>

namespace
{
JsonNode parse(const std::string & line)
{
	JsonParsingSettings settings;
	settings.mode = JsonParsingSettings::JsonFormatMode::JSON;
	settings.strict = true;
	return JsonNode(line.data(), line.size(), settings, "native session command");
}
JsonNode boundaryJson(const native_session::Boundary & boundary)
{
	JsonNode result;
	result["generation"].Integer() = boundary.token.generation;
	result["decision"].Integer() = boundary.token.decision;
	result["needs_action"].Bool() = boundary.needsAction;
	result["terminal_observed"].Bool() = boundary.terminalObserved;
	result["messages"].Vector() = boundary.messages;
	return result;
}
void timing(JsonNode & output, const native_session::SessionTimings & value)
{
	output["load_seconds"].Float() = value.loadSeconds;
	output["reset_seconds"].Float() = value.resetSeconds;
	output["advance_seconds_inference_excluded"].Float() = value.advanceSeconds;
	output["cleanup_seconds"].Float() = value.cleanupSeconds;
	output["episodes"].Integer() = value.episodes;
	output["steps"].Integer() = value.steps;
}
}

/// Native integration harness: explicit ordered commands, never an autonomous
/// policy or training loop. Responses are the existing adapter envelopes.
/// A completed script is NOT independent capture/replay acceptance.
int main(int argc, const char * argv[])
{
	if(argc != 5 || std::string(argv[1]) != "--commands" || std::string(argv[3]) != "--output")
	{
		std::cerr << "Usage: vcmisession --commands <commands.jsonl> --output <new-directory>\n";
		return 2;
	}
	const std::filesystem::path output(argv[4]);
	if(std::filesystem::exists(output)) { std::cerr << "Output already exists\n"; return 2; }
	std::filesystem::create_directories(output);
	std::ofstream receipts(output / "receipts.jsonl");
	JsonNode summary;
	summary["schema"].String() = "heroes-native-live-session-v1";
	summary["status"].String() = "incomplete";
	summary["replay_verified"].Bool() = false;
	summary["execution"].String() = "in-process live VCMI server/client; typed policy channel";
	summary["rng_boundary"].String() = "Inherited VCMI/Nullkiller/BattleAI RNG and thread scheduling; no cross-platform bit identity promise";
	int code = 0;
	{
		native_session::EngineSession session;
		try
		{
			std::ifstream input(argv[2]);
			if(!input) throw std::runtime_error("Cannot read commands");
			std::optional<native_session::Boundary> boundary;
			std::string line;
			int64_t sequence = 0;
			while(std::getline(input, line))
			{
				if(line.empty()) throw std::runtime_error("Empty native session command");
				const auto command = parse(line);
				const auto kind = command["kind"].String();
				JsonNode receipt;
				receipt["sequence"].Integer() = sequence++;
				receipt["kind"].String() = kind;
				if(kind == "load") session.load();
				else if(kind == "reset")
				{
					native_session::EpisodeConfig config;
					config.episode = command["episode"].String();
					config.map = command["map"].String();
					config.captureDirectory = command["capture_directory"].String();
					config.seed = command["seed"].Integer();
					config.learner = command["learner"].Integer();
					session.reset(config);
					boundary.reset();
				}
				else if(kind == "observe" || kind == "step")
				{
					const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
					if(kind == "step")
					{
						if(!boundary || !boundary->needsAction) throw std::runtime_error("step requires an observed decision");
						boundary = session.step(boundary->token, command["response"], deadline);
					}
					else boundary = session.observe(deadline);
					receipt["boundary"] = boundaryJson(*boundary);
				}
				else if(kind == "await_shutdown") session.waitForShutdown(std::chrono::steady_clock::now() + std::chrono::seconds(30));
				else if(kind == "close") { session.closeEpisode(); boundary.reset(); }
				else throw std::runtime_error("Unknown native session command");
				receipt["native_shutdown_complete"].Bool() = session.shutdownComplete();
				timing(receipt["timings"], session.timings());
				if(const auto peak = native_session_host::peakResidentBytes()) receipt["peak_process_resident_bytes"].Integer() = *peak;
				auto text = receipt.toCompactString();
				text.erase(std::remove(text.begin(), text.end(), '\n'), text.end());
				receipts << text << '\n';
				receipts.flush();
			}
			session.closeEpisode();
			summary["status"].String() = "commands_completed_unverified";
		}
		catch(const std::exception & error)
		{
			summary["status"].String() = "failed";
			summary["error"].String() = error.what();
			code = 3;
			session.closeEpisode();
		}
		timing(summary["timings"], session.timings());
	}
	if(const auto peak = native_session_host::peakResidentBytes()) summary["peak_process_resident_bytes"].Integer() = *peak;
	summary["peak_memory_scope"].String() = "Process high-water mark across the whole campaign, including retained resources and cleanup";
	std::ofstream(output / "summary.json") << summary.toString() << '\n';
	return code;
}
