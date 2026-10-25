// Process-lifecycle checks only; this is not a substitute game implementation.
// License: GNU General Public License v2.0 or later
#include "ForkEpisode.h"
#include <cassert>
#include <random>
#include <atomic>
#include <fstream>

int main(int argc, char ** argv)
{
	using namespace optimizer;
	using namespace std::chrono_literals;
	assert(argc == 2);
	std::mt19937 rng(42);
	const auto initial = rng;
	const auto expected = std::mt19937(42)();
	int state = 0;
	const int descriptor = open(argv[1], O_RDONLY);
	assert(descriptor >= 0);
	for(int episode = 0; episode != 100; ++episode)
	{
		const auto result = forkEpisode([&]()
		{
			if(state != 0 || rng() != expected || lseek(descriptor, 0, SEEK_CUR) != 0)
				return 1;
			state = 100;
			rng.discard(1000);
			char buffer[4];
			return read(descriptor, buffer, sizeof(buffer)) == 4 ? 0 : 2;
		}, 1s);
		assert(result.exitCode == 0 && result.signal == 0 && !result.timedOut);
		assert(state == 0 && rng == initial && lseek(descriptor, 0, SEEK_CUR) == 0);
	}
	assert(forkEpisode([]() { return 7; }, 1s).exitCode == 7);
	assert(forkEpisode([]() -> int { throw std::runtime_error("failure"); }, 1s).exitCode == 125);
	const auto crash = forkEpisode([]() { raise(SIGKILL); return 0; }, 1s);
	assert(crash.signal == SIGKILL && crash.exitCode == -1 && !crash.timedOut);
	const auto timeout = forkEpisode([&]()
	{
		lseek(descriptor, 4, SEEK_SET);
		for(;;) pause();
		return 0;
	}, 20ms);
	assert(timeout.timedOut && timeout.signal == SIGKILL && timeout.seconds < 1);
	assert(lseek(descriptor, 0, SEEK_CUR) == 0);
	assert(forkEpisode([]() { return 0; }, 1s).exitCode == 0);
	bool rejected = false;
	try { forkEpisode([]() { return 0; }, 0ms); }
	catch(const std::runtime_error &) { rejected = true; }
	assert(rejected);
	std::atomic<bool> stop = false;
	std::thread other([&]() { while(!stop) std::this_thread::yield(); });
	rejected = false;
	try { forkEpisode([]() { return 0; }, 1s); }
	catch(const std::runtime_error &) { rejected = true; }
	stop = true;
	other.join();
	assert(rejected);
	int status;
	assert(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD);
	close(descriptor);
	return 0;
}
