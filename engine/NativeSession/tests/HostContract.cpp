// Host ownership/lifecycle admission fixture; no game assets or simulated rules.
#include "../StdInc.h"
#include "../EngineSession.h"
#include <iostream>
namespace
{
template<class F> void rejects(F action)
{
	try { action(); } catch(const std::exception &) { return; }
	throw std::runtime_error("Missing expected native host admission failure");
}
}
int main()
{
	try
	{
		for(int episode = 0; episode < 100; ++episode)
		{
			native_session::EngineSession owner;
			rejects([] { native_session::EngineSession duplicate; });
			rejects([&] { owner.reset({}); });
			rejects([&] { owner.observe(std::chrono::steady_clock::now()); });
			rejects([&] { owner.step({}, JsonNode(), std::chrono::steady_clock::now()); });
			rejects([&] { owner.waitForShutdown(std::chrono::steady_clock::now()); });
			owner.closeEpisode();
			owner.closeEpisode();
			if(owner.shutdownComplete() || owner.timings().episodes)
				throw std::runtime_error("Unloaded session falsely reported a completed game");
		}
		std::cout << "100 exclusive native host lifetimes passed. No game state or resources loaded.\n";
		return 0;
	}
	catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
