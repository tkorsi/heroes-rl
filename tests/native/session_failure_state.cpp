// Production lifecycle admission only; no engine state, game assets or rules.
#include "SessionHealth.h"
#include <future>
#include <iostream>
#include <mutex>
#include <thread>

using namespace native_session_host;
void check(bool value) { if(!value) throw std::runtime_error("Session failure admission regression"); }
template<class F> void rejects(F action)
{
	try { action(); } catch(const std::runtime_error &) { return; } catch(const std::logic_error &) { return; }
	throw std::runtime_error("Failed session was admitted for reuse");
}
int main()
{
	try
	{
		for(int iteration = 0; iteration < 100; ++iteration)
		{
			SessionHealth health;
			ShutdownStatus shutdown;
			shutdown.record(1); // failed active episode, not yet closed
			bool mutatedNextEpisode = false;
			rejects([&] {
				health.prepareReset(true, [&] { if(shutdown.failed()) health.poison(); });
				mutatedNextEpisode = true;
			});
			check(health.failed() && !mutatedNextEpisode);
			bool cleanupRepeated = false;
			rejects([&] { health.prepareReset(true, [&] { cleanupRepeated = true; }); });
			check(!cleanupRepeated);

			SessionHealth lateHealth;
			ShutdownStatus late;
			std::mutex mutex;
			bool lateAccepted = false;
			std::promise<void> cleanupStarted;
			auto stopping = cleanupStarted.get_future();
			std::thread producer([&] {
				stopping.wait();
				std::lock_guard lock(mutex);
				lateAccepted = late.record(1); // real error reported after cancellation starts
			});
			rejects([&] {
				lateHealth.prepareReset(true, [&] {
					{ std::lock_guard lock(mutex); late.stopping = true; }
					cleanupStarted.set_value();
					producer.join();
					if(late.failed()) lateHealth.poison();
				});
			});
			check(lateAccepted && lateHealth.failed() && late.code == 1);
			check(!late.record(0) && late.code == 1); // success cannot overwrite failure

			SessionHealth clean;
			ShutdownStatus cancelled;
			clean.prepareReset(true, [&] {
				cancelled.stopping = true;
				check(!cancelled.record(0));
				if(cancelled.failed()) clean.poison();
			});
			clean.prepareReset(true, [] {});
			check(!clean.failed() && !cancelled.code); // intentional truncation remains reusable

			SessionHealth brokenCleanup;
			rejects([&] { brokenCleanup.prepareReset(true, [] { throw std::runtime_error("join failed"); }); });
			check(brokenCleanup.failed());
		}
		std::cout << "100 failure/reset, late-report, clean-cancel and cleanup-error checks passed; no gameplay\n";
		return 0;
	}
	catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
