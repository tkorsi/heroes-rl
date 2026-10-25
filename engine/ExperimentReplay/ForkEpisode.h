/*
 * ForkEpisode.h, part of the Heroes RL experiment tools for VCMI
 * License: GNU General Public License v2.0 or later
 */
#pragma once

#ifdef __linux__
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <stdexcept>
#include <thread>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/stat.h>

namespace optimizer
{
/// Experimental process reset. The parent must never create episode state.
/// This is deliberately sequential and refuses a multithreaded template.
struct EpisodeResult
{
	int exitCode;
	int signal;
	bool timedOut;
	double seconds;
};

inline void requireSingleThread()
{
	size_t count = 0;
	for(const auto & entry : std::filesystem::directory_iterator("/proc/self/task"))
	{
		(void)entry;
		++count;
	}
	if(count != 1)
		throw std::runtime_error("Fork template must have exactly one thread");
}

/// fork shares open-file descriptions, including archive seek positions. Restore
/// each seekable inherited read-only descriptor after the child, even on timeout/crash.
/// This does not isolate external file writes: use immutable assets and separate
/// output paths; never use a live gameplay process as the template.
inline std::map<int, off_t> descriptorPositions()
{
	std::map<int, off_t> positions;
	for(const auto & entry : std::filesystem::directory_iterator("/proc/self/fd"))
	{
		const int fd = std::stoi(entry.path().filename().string());
		struct stat info{};
		if(fstat(fd, &info) != 0 || !S_ISREG(info.st_mode))
			continue;
		const auto offset = lseek(fd, 0, SEEK_CUR);
		if(fd > STDERR_FILENO && offset != -1 && (fcntl(fd, F_GETFL) & O_ACCMODE) == O_RDONLY)
			positions.emplace(fd, offset);
	}
	return positions;
}

inline EpisodeResult forkEpisode(const std::function<int()> & run, std::chrono::milliseconds limit)
{
	if(limit.count() <= 0)
		throw std::runtime_error("Episode deadline must be positive");
	requireSingleThread();
	const auto positions = descriptorPositions();
	std::fflush(nullptr);
	const auto start = std::chrono::steady_clock::now();
	const pid_t parent = getpid();
	const pid_t child = fork();
	if(child < 0)
		throw std::runtime_error("Cannot fork episode");
	if(child == 0)
	{
		if(prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != parent)
			_exit(125);
		if(setpgid(0, 0) != 0)
			_exit(125);
		int code = 125;
		try { code = run(); }
		catch(...) { code = 125; }
		std::fflush(nullptr);
		// No inherited global destructors/settings saves in the episode process.
		_exit(code >= 0 && code <= 125 ? code : 125);
	}
	// Either side may win this race. The child sets its own group before work.
	setpgid(child, child);
	int status = 0;
	bool timedOut = false;
	for(;;)
	{
		const auto waited = waitpid(child, &status, WNOHANG);
		if(waited == child)
			break;
		if(waited < 0 && errno != EINTR)
			throw std::runtime_error("Cannot reap episode");
		if(std::chrono::steady_clock::now() - start >= limit)
		{
			timedOut = true;
			kill(-child, SIGKILL);
			kill(child, SIGKILL);
			while(waitpid(child, &status, 0) < 0)
				if(errno != EINTR)
					throw std::runtime_error("Cannot reap expired episode");
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	// Descendants must not outlive even a normally exiting episode leader.
	kill(-child, SIGKILL);
	for(const auto & [fd, offset] : positions)
		if(lseek(fd, offset, SEEK_SET) != offset)
			throw std::runtime_error("Cannot restore inherited descriptor position");
	return {WIFEXITED(status) ? WEXITSTATUS(status) : -1,
		WIFSIGNALED(status) ? WTERMSIG(status) : 0, timedOut,
		std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
}
}
#endif
