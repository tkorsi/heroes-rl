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
#include <iostream>
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

// Reserved in episode callbacks. Ordinary descendants inherit this handler and
// remain in the episode group; callbacks must not block/replace it or escape groups.
inline void parentDisappeared(int)
{
	kill(-getpgrp(), SIGKILL);
	_exit(125);
}

inline EpisodeResult forkEpisode(const std::function<int()> & run, std::chrono::milliseconds limit,
	const std::function<void(pid_t)> & started = [](pid_t) {})
{
	if(limit.count() <= 0)
		throw std::runtime_error("Episode deadline must be positive");
	requireSingleThread();
	const auto positions = descriptorPositions();
	std::cout.flush();
	std::cerr.flush();
	std::fflush(nullptr);
	const auto start = std::chrono::steady_clock::now();
	const pid_t parent = getpid();
	const pid_t child = fork();
	if(child < 0)
		throw std::runtime_error("Cannot fork episode");
	if(child == 0)
	{
		if(setpgid(0, 0) != 0)
			_exit(125);
		struct sigaction action{};
		action.sa_handler = parentDisappeared;
		sigemptyset(&action.sa_mask);
		sigset_t unblock;
		sigemptyset(&unblock);
		sigaddset(&unblock, SIGUSR2);
		if(sigaction(SIGUSR2, &action, nullptr) != 0 || sigprocmask(SIG_UNBLOCK, &unblock, nullptr) != 0
			|| prctl(PR_SET_PDEATHSIG, SIGUSR2) != 0)
			_exit(125);
		if(getppid() != parent)
			parentDisappeared(SIGUSR2);
		int code = 125;
		try { code = run(); }
		catch(...) { code = 125; }
		std::cout.flush();
		std::cerr.flush();
		std::fflush(nullptr);
		// No inherited global destructors/settings saves in the episode process.
		_exit(code >= 0 && code <= 125 ? code : 125);
	}
	// Either side may win this race. The child sets its own group before work.
	setpgid(child, child);
	int status = 0;
	bool timedOut = false;
	try
	{
		started(child);
		for(;;)
		{
			siginfo_t info{};
			const auto waited = waitid(P_PID, child, &info, WEXITED | WNOHANG | WNOWAIT);
			if(waited == 0 && info.si_pid == child)
			{
				// Keep the leader unreaped until group cleanup: its PID cannot be
				// recycled into an unrelated process group between wait and kill.
				kill(-child, SIGKILL);
				while(waitpid(child, &status, 0) < 0)
					if(errno != EINTR)
						throw std::runtime_error("Cannot reap episode");
				break;
			}
			if(waited < 0 && errno != EINTR)
				throw std::runtime_error("Cannot observe episode exit");
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
		// The service is a subreaper: ordinary orphaned descendants are ours.
		// Reap this episode's group only, with a fixed additional cleanup bound.
		const auto cleanupEnd = std::chrono::steady_clock::now() + std::chrono::seconds(1);
		for(;;)
		{
			int descendantStatus;
			const auto descendant = waitpid(-child, &descendantStatus, WNOHANG);
			if(descendant > 0 || (descendant < 0 && errno == EINTR)) continue;
			if(descendant < 0 && errno == ECHILD) break;
			if(descendant < 0 || std::chrono::steady_clock::now() >= cleanupEnd)
				throw std::runtime_error("Episode group cleanup incomplete");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	catch(...)
	{
		kill(-child, SIGKILL);
		kill(child, SIGKILL);
		while(waitpid(child, &status, 0) < 0 && errno == EINTR) {}
		for(const auto & [fd, offset] : positions)
			lseek(fd, offset, SEEK_SET);
		throw;
	}
	for(const auto & [fd, offset] : positions)
		if(lseek(fd, offset, SEEK_SET) != offset)
			throw std::runtime_error("Cannot restore inherited descriptor position");
	return {WIFEXITED(status) ? WEXITSTATUS(status) : -1,
		WIFSIGNALED(status) ? WTERMSIG(status) : 0, timedOut,
		std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()};
}
}
#endif
