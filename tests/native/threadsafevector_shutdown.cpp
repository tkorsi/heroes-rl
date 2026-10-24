// Concurrency regression for the exact ThreadSafeVector template extracted from
// prepared VCMI Client.h by test_native_shutdown.py. This is not a game fixture.
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

// Provide the template's ordinary container helpers and exception type without
// loading the game engine, renderer, networking, or any AI module.
namespace vstd
{
template<typename Container, typename Value>
auto find(Container & container, const Value & value)
{
	return std::find(container.begin(), container.end(), value);
}

template<typename Container, typename Value>
bool contains(Container & container, const Value & value)
{
	return find(container, value) != container.end();
}
}

class TerminationRequestedException : public std::exception {};

#include "threadsafevector_under_test.h"

void require(bool condition, const char * message)
{
	if(!condition)
		throw std::runtime_error(message);
}

void runRegression()
{
	// The regular acknowledgement path must still wake a waiting sender.
	{
		ThreadSafeVector<int> queue;
		queue.pushBack(7);
		bool completed = false;
		std::thread sender([&]() { queue.waitWhileContains(7); completed = true; });
		require(queue.tryRemovingElement(7), "normal acknowledgement was lost");
		sender.join();
		require(completed, "normal sender did not complete");
		require(!queue.tryRemovingElement(7), "duplicate acknowledgement was accepted");
	}

	// This exact ordering previously inserted an unanswerable request in Release
	// builds: assert(!isTerminating) disappeared and the wait ignored termination.
	{
		ThreadSafeVector<int> queue;
		queue.requestTermination();
		bool rejected = false;
		try { queue.pushBack(8); }
		catch(const TerminationRequestedException &) { rejected = true; }
		require(rejected, "request was inserted after termination");
		require(!queue.tryRemovingElement(8), "late acknowledgement changed a terminated queue");
		bool interrupted = false;
		try { queue.waitWhileContains(8); }
		catch(const TerminationRequestedException &) { interrupted = true; }
		require(interrupted, "wait begun after termination did not interrupt");
		queue.requestTermination(); // Repeated cleanup remains safe.
	}

	// Exercise both lock acquisition orders for a sender racing cleanup. Every
	// sender must finish by observing termination, including one already waiting.
	for(int iteration = 0; iteration < 512; ++iteration)
	{
		ThreadSafeVector<int> queue;
		std::atomic<bool> started = false;
		bool interrupted = false;
		std::thread sender([&]() {
			started = true;
			try
			{
				queue.pushBack(iteration);
				queue.waitWhileContains(iteration);
			}
			catch(const TerminationRequestedException &) { interrupted = true; }
		});
		while(!started)
			std::this_thread::yield();
		if(iteration % 2)
			std::this_thread::yield();
		queue.requestTermination();
		sender.join();
		require(interrupted, "racing sender did not observe termination");
	}
	std::cout << "ThreadSafeVector shutdown regression passed (512 racing senders).\n";
}

int main()
{
	try
	{
		runRegression();
		return 0;
	}
	catch(const std::exception & error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
