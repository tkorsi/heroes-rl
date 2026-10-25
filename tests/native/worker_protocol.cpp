// Actual production protocol with process/transport callbacks. No game simulation.
// License: GNU General Public License v2.0 or later
#include "WorkerProtocol.h"
#include <unistd.h>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <thread>

int main(int argc, char ** argv)
{
	if(argc != 3) return 2;
	const int output = dup(STDOUT_FILENO);
	dup2(STDERR_FILENO, STDOUT_FILENO);
	int state = 0;
	return optimizer::serveWorker(STDIN_FILENO, output, std::chrono::milliseconds(std::stoi(argv[2])), []()
	{
		if(const auto delay = std::getenv("HARNESS_INIT_MS"))
			std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(delay)));
	}, [&](const std::vector<std::string> & args)
	{
		if(state != 0) return 9;
		state = 1;
		std::string mode, report;
		for(size_t i = 0; i + 1 < args.size(); ++i)
		{
			if(args[i] == "--log") mode = args[i + 1];
			if(args[i] == "--report") report = args[i + 1];
		}
		if(mode == "crash") raise(SIGKILL);
		if(mode == "throw") throw std::runtime_error("Callback failure");
		if(mode == "fail") return 7;
		if(mode == "descendant" || mode == "descendant_hang" || mode == "descendant_crash")
		{
			const auto descendant = fork();
			if(descendant < 0) return 8;
			if(descendant == 0) for(;;) pause();
			std::ofstream file(report);
			file << descendant << '\n';
			file.close();
		}
		if(mode == "descendant_crash") raise(SIGKILL);
		if(mode == "hang" || mode == "descendant_hang") for(;;) pause();
		return 0;
	});
}
