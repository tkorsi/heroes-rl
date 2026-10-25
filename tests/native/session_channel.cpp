// Production native transport/reset contract tests. No game rules or game data.
#include "nativesession/PolicyTransport.h"
#include "lib/callback/ExperimentHooks.h"
#include "lib/callback/CGameInterface.h"
#include <atomic>
#include <future>
#include <iostream>
#include <thread>
#include <stdexcept>

using namespace native_session;
using namespace std::chrono_literals;
namespace
{
void check(bool value, const char * reason) { if(!value) throw std::runtime_error(reason); }
template<class F> void rejects(F action)
{
	try { action(); } catch(const std::exception &) { return; }
	throw std::runtime_error("Expected rejection");
}
JsonNode message(const std::string & type, int marker = 0)
{
	JsonNode result;
	result["type"].String() = type;
	result["fixture_marker"].Integer() = marker;
	return result;
}
Deadline deadline() { return std::chrono::steady_clock::now() + 2s; }
}
int main()
{
	try
	{
		std::shared_ptr<InProcessPolicy> previous;
		BoundaryToken old{};
		for(uint64_t episode = 1; episode <= 100; ++episode)
		{
			auto policy = std::make_shared<InProcessPolicy>(episode);
			installPolicy(0, "fixture-" + std::to_string(episode), policy);
			rejects([&] { claimPolicy(0, "wrong-episode"); });
			check(claimPolicy(0, "fixture-" + std::to_string(episode)) == policy, "Cross-library binding differs");
			rejects([&] { claimPolicy(0, "fixture-" + std::to_string(episode)); });
			auto adapter = std::async(std::launch::async, [&]
			{
				policy->send(message("fixture-ready", static_cast<int>(episode)));
				policy->send(message("fixture-visible-events", static_cast<int>(episode)));
				auto response = policy->receive(deadline());
				check(response["fixture_marker"].Integer() == static_cast<int64_t>(episode), "Stale response crossed reset");
				policy->send(message("fixture-result", static_cast<int>(episode)));
				policy->send(message("terminal", static_cast<int>(episode)));
			});
			const auto boundary = policy->observe(deadline());
			check(boundary.needsAction && boundary.messages.size() == 2, "Lost or reordered fixture observations");
			check(policy->observe(deadline()).messages == boundary.messages, "observe must be idempotent");
			if(previous)
			{
				rejects([&] { previous->submit(old, message("fixture-response")); });
				rejects([&] { policy->submit(old, message("fixture-response")); });
			}
			policy->submit(boundary.token, message("fixture-response", static_cast<int>(episode)));
			rejects([&] { policy->submit(boundary.token, message("fixture-response")); });
			const auto terminal = policy->observe(deadline());
			check(terminal.terminalObserved && !terminal.needsAction && terminal.messages.size() == 2, "Incorrect terminal boundary");
			adapter.get();
			removePolicy(0);
			check(policy->cancelled(), "Removed policy not closed");
			check(!claimPolicy(0, "fixture"), "Binding leaked after cleanup");
			previous = policy;
			old = boundary.token;
			// These are the actual diagnostic counters used by live packet hooks.
			setExperimentEvidenceHooks(nullptr);
			resetNativeSessionEvidence();
			noteExperimentPackApplied(12);
			check(experimentPackPosition().index == 1, "Packet counter leaked across reset");
		}
		{
			auto policy = std::make_shared<InProcessPolicy>(101);
			auto adapter = std::async(std::launch::async, [&] { rejects([&] { policy->receive(deadline()); }); });
			policy->observe(deadline());
			policy->close();
			adapter.get();
			rejects([&] { policy->observe(deadline()); });
		}
		{
			auto policy = std::make_shared<InProcessPolicy>(102);
			auto adapter = std::async(std::launch::async, [&] { rejects([&] { policy->receive(deadline()); }); });
			const auto pending = policy->observe(deadline());
			policy->send(message("terminal")); // terminal supersedes an observed pending action
			check(!policy->observe(deadline()).needsAction, "Terminal left a live decision token");
			check(!policy->submit(pending.token, message("fixture-response")), "Superseded response was applied");
			rejects([&] { policy->submit(pending.token, message("fixture-response")); });
			policy->close();
			adapter.get();
		}
		{
			InProcessPolicy policy(103, 1);
			policy.send(message("fixture"));
			rejects([&] { policy.send(message("fixture-overflow")); });
			rejects([&] { policy.observe(deadline()); });
		}
		{
			InProcessPolicy policy(104);
			rejects([&] { policy.receive(std::chrono::steady_clock::now() + 10ms); });
			rejects([&] { policy.observe(deadline()); });
		}
		{
			InProcessPolicy left(105), right(106);
			left.send(message("terminal", 7));
			right.send(message("terminal", 8));
			check(left.observe(deadline()).messages[0]["fixture_marker"].Integer() == 7, "Cross-player leak");
			check(right.observe(deadline()).messages[0]["fixture_marker"].Integer() == 8, "Cross-player leak");
		}
		{
			InProcessPolicy policy(107);
			auto envelope = message("contextual");
			JsonNode terminal;
			terminal["kind"].String() = "terminal";
			envelope["records"].Vector().push_back(terminal);
			policy.send(envelope);
			check(policy.observe(deadline()).terminalObserved, "Contextual terminal was not recognized");
		}
		std::cout << "100 native channel generations, cancellation, deadlines, terminal, capacity and evidence reset passed; no gameplay simulated\n";
		return 0;
	}
	catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
