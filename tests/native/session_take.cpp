// Incremental in-process delivery contract (take/submit/drain). No game rules or data.
#include "nativesession/PolicyTransport.h"
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace native_session;
using namespace std::chrono_literals;
namespace
{
void check(bool value, const char * reason) { if(!value) throw std::runtime_error(reason); }
template<class F> std::string rejects(F action)
{
	try { action(); } catch(const std::exception & error) { return error.what(); }
	throw std::runtime_error("Expected rejection");
}
JsonNode message(const std::string & type, int marker = 0)
{
	JsonNode result;
	result["type"].String() = type;
	result["fixture_marker"].Integer() = marker;
	return result;
}
JsonNode terminalBatch(int marker)
{
	JsonNode result = message("contextual", marker);
	JsonNode record;
	record["kind"].String() = "terminal";
	result["records"].Vector().push_back(record);
	return result;
}
Deadline soon(std::chrono::milliseconds value = 2000ms) { return std::chrono::steady_clock::now() + value; }
std::vector<int> markers(const Delivery & delivery)
{
	std::vector<int> result;
	for(const auto & item : delivery.messages) result.push_back(static_cast<int>(item["fixture_marker"].Integer()));
	return result;
}
}
int main()
{
	try
	{
		// Each message is delivered exactly once, in order; a boundary offers one submit.
		{
			InProcessPolicy policy(1);
			policy.send(message("ready", 1));
			auto partial = policy.take(soon(50ms), false);
			check(markers(partial) == std::vector<int>{1} && !partial.needsAction && !partial.timedOut, "message before boundary");
			check(policy.take(soon(20ms), false).timedOut, "nothing new times out");
			std::promise<JsonNode> answer;
			std::thread worker([&] {
				policy.send(message("contextual", 2));
				answer.set_value(policy.receive(soon()));
			});
			auto boundary = policy.take(soon(), true);
			check(markers(boundary) == std::vector<int>{2} && boundary.needsAction, "decision boundary");
			check(boundary.token == BoundaryToken{1, 1}, "first decision token");
			auto again = policy.take(soon(20ms), true);
			check(again.messages.empty() && !again.needsAction && again.timedOut, "a boundary is offered once");
			rejects([&] { policy.submit({1, 2}, message("context_action")); });
			policy.submit(boundary.token, message("context_action", 3));
			rejects([&] { policy.submit(boundary.token, message("context_action", 4)); });
			check(answer.get_future().get()["fixture_marker"].Integer() == 3, "submitted response received");
			worker.join();
		}
		// The terminal wakes a pending receive as a typed, benign cancellation and
		// later permitted envelopes are queued behind it, never rejected.
		{
			InProcessPolicy policy(2);
			std::promise<std::string> outcome;
			std::thread worker([&] {
				try { policy.receive(soon()); outcome.set_value("response"); }
				catch(const PolicyTransportTerminal &) { outcome.set_value("terminal"); }
				catch(const std::exception & error) { outcome.set_value(error.what()); }
			});
			auto offered = policy.take(soon(), true);
			check(offered.needsAction, "waiting receive offered");
			policy.send(terminalBatch(5));
			check(outcome.get_future().get() == "terminal", "terminal superseded the receive");
			worker.join();
			policy.send(message("contextual", 6));
			auto end = policy.take(soon(), true);
			check(end.terminalObserved && !end.needsAction && end.failure.empty(), "terminal reported");
			check(markers(end) == std::vector<int>{5, 6}, "terminal and later envelope delivered once");
			rejects([&] { policy.submit(end.token, message("context_action")); });
			auto later = policy.take(soon(20ms), true);
			check(later.messages.empty() && later.terminalObserved && !later.needsAction, "terminal does not wake twice");
			policy.send(message("contextual", 7));
			policy.close();
			check(markers(Delivery{{}, policy.drain()}) == std::vector<int>{7}, "drain returns the undelivered tail");
			rejects([&] { policy.send(message("contextual", 8)); });
		}
		// An ExplorerRL error record is delivered with its prefix and the first
		// failure reason is kept; a later report never replaces it.
		{
			InProcessPolicy policy(3);
			policy.send(message("contextual", 9));
			JsonNode error = message("error", 10);
			error["message"].String() = "unsupported fixture interaction";
			policy.send(error);
			policy.fail("later host failure");
			auto failed = policy.take(soon(), true);
			check(markers(failed) == std::vector<int>{9, 10}, "failure prefix and error record delivered");
			check(failed.failure == "ExplorerRL: unsupported fixture interaction", "first failure kept");
			check(policy.failure() == failed.failure && policy.failed(), "failure stays first");
			rejects([&] { policy.receive(soon()); });
		}
		// A receive timeout is a failure, never a cancellation.
		{
			InProcessPolicy policy(4);
			auto reason = rejects([&] { policy.receive(soon(20ms)); });
			check(reason == "Native policy response timeout" && policy.failure() == reason, "timeout failure");
			check(policy.take(soon(20ms), true).failure == reason, "timeout reported to the host");
		}
		// High water: a long opponent turn wakes the host before send() overflows.
		{
			InProcessPolicy policy(5, 8);
			std::thread producer([&] { for(int index = 0; index < 4; ++index) policy.send(message("contextual", 20 + index)); });
			producer.join();
			auto batch = policy.take(soon(), true);
			check(batch.messages.size() == 4 && !batch.timedOut && !batch.needsAction, "half-full queue handed over");
			for(int index = 0; index < 8; ++index) policy.send(message("contextual", 30 + index));
			auto overflow = rejects([&] { policy.send(message("contextual", 40)); });
			check(overflow == "Native policy message queue overflow" && policy.failure() == overflow, "overflow is a failure");
			check(policy.take(soon(), true).messages.size() == 8, "accepted messages survive overflow");
		}
		// Responses keep the socket backend's 64 KiB admission.
		{
			InProcessPolicy policy(6);
			std::thread worker([&] { try { policy.receive(soon(500ms)); } catch(...) {} });
			auto offered = policy.take(soon(), true);
			JsonNode large = message("context_action");
			large["padding"].String() = std::string(70000, 'x');
			rejects([&] { policy.submit(offered.token, large); });
			policy.close();
			worker.join();
		}
		std::cout << "incremental native channel contract passed; no gameplay simulated\n";
		return 0;
	}
	catch(const std::exception & error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
