// Actual production channel cancellation/error ordering, no game state.
#include "PolicyTransport.h"
#include <future>
#include <iostream>
#include <thread>

using namespace native_session;
using namespace std::chrono_literals;
void check(bool value) { if(!value) throw std::runtime_error("Channel failure classification regression"); }
template<class F> void realFailure(F action)
{
	try { action(); }
	catch(const PolicyTransportCancelled &) { throw std::runtime_error("Genuine error became cancellation"); }
	catch(const std::runtime_error &) { return; }
	throw std::runtime_error("Missing expected channel failure");
}
int main()
{
	try
	{
		for(uint64_t iteration = 1; iteration <= 100; ++iteration)
		{
			auto policy = std::make_shared<InProcessPolicy>(iteration);
			std::promise<void> continueProducer;
			auto resume = continueProducer.get_future();
			std::thread producer([&] {
				JsonNode error;
				error["type"].String() = "error";
				error["message"].String() = "Contextual response identity mismatch";
				policy->send(error);
				resume.wait(); // native failure reporting can happen after controller cleanup starts
				policy->fail("late native failure report");
			});
			realFailure([&] { policy->observe(std::chrono::steady_clock::now() + 2s); });
			policy->close();
			continueProducer.set_value();
			producer.join();
			check(policy->cancelled() && policy->failed());
			realFailure([&] { policy->receive(std::chrono::steady_clock::now()); });
			realFailure([&] { policy->observe(std::chrono::steady_clock::now()); });

			InProcessPolicy clean(1000 + iteration);
			clean.close();
			check(clean.cancelled() && !clean.failed());
			bool cancelled = false;
			try { clean.receive(std::chrono::steady_clock::now()); }
			catch(const PolicyTransportCancelled &) { cancelled = true; }
			check(cancelled);
		}
		InProcessPolicy timeout(2001);
		realFailure([&] { timeout.receive(std::chrono::steady_clock::now() + 1ms); });
		timeout.close();
		check(timeout.failed());
		realFailure([&] { timeout.observe(std::chrono::steady_clock::now()); });
		InProcessPolicy overflow(2002, 1);
		JsonNode message;
		message["type"].String() = "fixture";
		overflow.send(message);
		realFailure([&] { overflow.send(message); });
		overflow.close();
		check(overflow.failed());
		std::cout << "100 error-before-close channels plus timeout/overflow preserve failure; clean close is typed cancellation\n";
		return 0;
	}
	catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
