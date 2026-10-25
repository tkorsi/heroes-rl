/* License: GNU General Public License v2.0 or later */
#include "StdInc.h"
#include "WorkerProtocol.h"
#include "ForkEpisode.h"
#include "../lib/json/JsonNode.h"
#ifdef __linux__
#include <poll.h>

namespace optimizer
{
namespace
{
bool readLine(int fd, std::string & line)
{
	line.clear();
	char byte;
	for(;;)
	{
		const auto count = read(fd, &byte, 1);
		if(count < 0 && errno == EINTR) continue;
		if(count < 0) throw std::runtime_error("Cannot read worker request");
		if(count == 0)
		{
			if(!line.empty()) throw std::runtime_error("Partial request at EOF");
			return false;
		}
		if(byte == '\n') return true;
		if(line.size() == 65536) throw std::runtime_error("Request exceeds 65536 bytes");
		line += byte;
	}
}

void send(int fd, JsonNode node)
{
	node["schema"].String() = WORKER_SCHEMA;
	auto line = node.toCompactString();
	line.erase(std::remove_if(line.begin(), line.end(), [](char c) { return c == '\n' || c == '\r'; }), line.end());
	line += '\n';
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
	size_t offset = 0;
	while(offset < line.size())
	{
		const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
		if(left <= 0) throw std::runtime_error("Worker output deadline");
		pollfd poller{fd, POLLOUT, 0};
		if(poll(&poller, 1, static_cast<int>(left)) < 0 && errno == EINTR) continue;
		const auto count = write(fd, line.data() + offset, line.size() - offset);
		if(count < 0 && (errno == EINTR || errno == EAGAIN)) continue;
		if(count <= 0) throw std::runtime_error("Worker output pipe closed");
		offset += count;
	}
}

std::vector<std::string> arguments(const JsonNode & node)
{
	if(!node.isVector()) throw std::runtime_error("args must be an array");
	std::vector<std::string> result;
	std::set<std::string> seen, outputs;
	const std::set<std::string> values{"--log", "--report", "--observations", "--trace", "--contextual"};
	for(size_t i = 0; i < node.Vector().size(); ++i)
	{
		if(node[i].getType() != JsonNode::JsonType::DATA_STRING || !seen.insert(node[i].String()).second)
			throw std::runtime_error("Job options must be unique strings");
		const auto name = node[i].String();
		result.push_back(name);
		if(values.count(name))
		{
			if(++i >= node.Vector().size() || node[i].getType() != JsonNode::JsonType::DATA_STRING
				|| node[i].String().empty() || node[i].String().find('\0') != std::string::npos)
				throw std::runtime_error("Missing or invalid job option value");
			result.push_back(node[i].String());
			if(name == "--report" || name == "--trace")
			{
				const auto path = boost::filesystem::absolute(result.back()).lexically_normal().string();
				if(boost::filesystem::exists(path) || !outputs.insert(path).second)
					throw std::runtime_error("Worker output exists or collides");
			}
		}
		else if(name != "--prefix") throw std::runtime_error("Unsupported job option");
	}
	if(!seen.count("--log") || !seen.count("--report")) throw std::runtime_error("Job needs --log and --report");
	return result;
}
}

int serveWorker(int input, int output, std::chrono::milliseconds limit,
	const std::function<void()> & initialize,
	const std::function<int(const std::vector<std::string> &)> & replay)
{
	try
	{
		if(limit.count() <= 0 || limit.count() > 3600000) throw std::runtime_error("Invalid job deadline");
		const auto supervisor = getppid();
		signal(SIGTERM, SIG_DFL);
		if(supervisor <= 1 || prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || prctl(PR_SET_CHILD_SUBREAPER, 1) != 0)
			throw std::runtime_error("Cannot bind worker to supervising parent");
		if(getppid() != supervisor) return 125;
		signal(SIGPIPE, SIG_IGN);
		if(fcntl(output, F_SETFL, fcntl(output, F_GETFL) | O_NONBLOCK) < 0)
			throw std::runtime_error("Cannot bound worker output");
		requireSingleThread();
		const auto start = std::chrono::steady_clock::now();
		initialize();
		requireSingleThread();
		JsonNode ready;
		ready["kind"].String() = "ready";
		ready["load_seconds"].Float() = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
		send(output, ready);
		int64_t next = 0;
		std::string line;
		while(readLine(input, line))
		{
			JsonParsingSettings parsing;
			parsing.mode = JsonParsingSettings::JsonFormatMode::JSON;
			parsing.strict = true;
			JsonNode job(reinterpret_cast<const std::byte *>(line.data()), line.size(), parsing, "worker request");
			if(!job.isStruct() || job.Struct().size() != 4 || job["schema"].getType() != JsonNode::JsonType::DATA_STRING
				|| job["schema"].String() != WORKER_SCHEMA || job["kind"].getType() != JsonNode::JsonType::DATA_STRING
				|| job["kind"].String() != "replay" || job["job_id"].getType() != JsonNode::JsonType::DATA_INTEGER
				|| job["job_id"].Integer() != next || !job.Struct().count("args"))
				throw std::runtime_error("Invalid worker schema, fields or ordered job_id");
			const auto args = arguments(job["args"]);
			const auto result = forkEpisode([&]()
			{
				close(input);
				close(output);
				return replay(args);
			}, limit, [&](pid_t pid)
			{
				JsonNode receipt;
				receipt["kind"].String() = "started";
				receipt["job_id"].Integer() = next;
				receipt["pid"].Integer() = pid;
				send(output, receipt);
			});
			JsonNode receipt;
			receipt["kind"].String() = "completed";
			receipt["job_id"].Integer() = next++;
			receipt["exit_code"].Integer() = result.exitCode;
			receipt["signal"].Integer() = result.signal;
			receipt["timed_out"].Bool() = result.timedOut;
			receipt["seconds"].Float() = result.seconds;
			receipt["cleanup_complete"].Bool() = true;
			send(output, receipt);
		}
		JsonNode closed;
		closed["kind"].String() = "closed";
		closed["jobs"].Integer() = next;
		send(output, closed);
		return 0;
	}
	catch(const std::exception & error)
	{
		std::cerr << "Worker protocol failure: " << error.what() << '\n';
		return 3;
	}
}
}
#endif
