// Legacy backend compatibility/cancellation fixture, not a game protocol peer.
#include "AI/ExplorerRL/SocketPolicyTransport.h"
#include <future>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;
int main(int argc, char ** argv)
{
	try
	{
		if(argc != 3) return 2;
		auto endpoint = makeSocketPolicyTransport(std::stoi(argv[1]), 2);
		JsonNode message;
		message["fixture"].String() = "line one\nline two";
		const auto stats = endpoint->send(message);
		if(stats.wireBytes <= 0) throw std::runtime_error("Missing wire metric");
		const std::string mode(argv[2]);
		if(mode == "normal")
		{
			const auto one = endpoint->receive(std::chrono::steady_clock::now() + 2s);
			const auto two = endpoint->receive(std::chrono::steady_clock::now() + 2s);
			if(one["fixture"].Integer() != 1 || two["fixture"].Integer() != 2)
				throw std::runtime_error("Fragmented/coalesced records changed");
		}
		else if(mode == "cancel")
		{
			auto reader = std::async(std::launch::async, [&]
			{
				try { endpoint->receive(std::chrono::steady_clock::now() + 2s); }
				catch(const std::runtime_error &) { return; }
				throw std::runtime_error("Cancelled receive unexpectedly completed");
			});
			endpoint->close();
			if(reader.wait_for(500ms) != std::future_status::ready)
				throw std::runtime_error("Socket shutdown did not wake receive");
			reader.get();
		}
		else if(mode == "oversized")
		{
			try { endpoint->receive(std::chrono::steady_clock::now() + 2s); }
			catch(const std::runtime_error &) { std::cout << "oversized response rejected\n"; return 0; }
			throw std::runtime_error("Oversized response admitted");
		}
		else return 2;
		std::cout << "Legacy socket fixture passed\n";
		return 0;
	}
	catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
