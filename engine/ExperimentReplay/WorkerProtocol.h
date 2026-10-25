/* License: GNU General Public License v2.0 or later */
#pragma once
#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace optimizer
{
inline constexpr const char * WORKER_SCHEMA = "heroes-replay-worker-v2";
/// Owns wire parsing, ordering and process reset. Callbacks supply initialization
/// and replay only; the native integration harness uses this same service.
int serveWorker(int input, int output, std::chrono::milliseconds limit,
	const std::function<void()> & initialize,
	const std::function<int(const std::vector<std::string> &)> & replay);
}
