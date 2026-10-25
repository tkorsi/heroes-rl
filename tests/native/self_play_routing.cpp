// Exercise exact production CClient routing and terminal methods extracted by
// test_native_self_play.py. This does not pretend to be a playable game fixture.
#include <compare>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

class PlayerColor
{
	int value;

public:
	explicit PlayerColor(int value) : value(value) {}
	int getNum() const { return value; }
	auto operator<=>(const PlayerColor &) const = default;
};

struct Logger
{
	int terminalMessages = 0;
	void info(const char *, int, const char *) { ++terminalMessages; }
};
Logger logger;
Logger * logAi = &logger;

bool contextualMode = false;
bool contextualEvidenceEnabled() { return contextualMode; }

class CClient
{
	std::set<PlayerColor> explorerTerminalsReceived;
	std::map<PlayerColor, bool> explorerTerminalOutcomes;
	std::map<PlayerColor, bool> builtinNativeTerminalOutcomes;

public:
	std::optional<PlayerColor> configuredExplorerPlayer() const;
	std::set<PlayerColor> configuredExplorerPlayers() const;
	bool recordExplorerTerminal(PlayerColor player, bool won);
	bool recordBuiltinNativeTerminal(PlayerColor player, bool won);
};

#include "self_play_under_test.h"

void require(bool condition, const char * message)
{
	if(!condition)
		throw std::runtime_error(message);
}

void configure(const char * singular, const char * plural)
{
	if(singular)
		setenv("EXPLORER_RL_PLAYER", singular, 1);
	else
		unsetenv("EXPLORER_RL_PLAYER");
	if(plural)
		setenv("EXPLORER_RL_PLAYERS", plural, 1);
	else
		unsetenv("EXPLORER_RL_PLAYERS");
}

void requireRejected(const char * singular, const char * plural)
{
	configure(singular, plural);
	bool rejected = false;
	try { CClient().configuredExplorerPlayers(); }
	catch(const std::runtime_error &) { rejected = true; }
	require(rejected, "invalid or ambiguous player routing was accepted");
}

void runRegression()
{
	unsetenv("EXPLORER_RL_BUILTIN_OPPONENT");
	unsetenv("EXPLORER_RL_INTERFACE");
	configure(nullptr, nullptr);
	require(experimentBuiltinLearner() == -1, "legacy mode acquired a mixed controller");
	require(CClient().configuredExplorerPlayers().empty(), "built-in mode acquired an external player");
	require(!CClient().configuredExplorerPlayer(), "built-in mode acquired a singular player");
	for(int player = 0; player < 8; ++player)
	{
		const auto value = std::to_string(player);
		configure(value.c_str(), nullptr);
		require(CClient().configuredExplorerPlayer() == PlayerColor(player), "singular routing changed");
		require(CClient().configuredExplorerPlayers() == std::set{PlayerColor(player)}, "singular set mismatch");
	}
	for(const auto invalid : {"", "8", "-1", "+0", " 0", "0 ", "00", "0,1"})
		requireRejected(invalid, nullptr);
	for(const auto invalid : {"", ",0", "0,", "0,,1", "0,0", "8,1", "0 1", "0;1", "0, 1", "0,-1", "0,+1", "0,8"})
		requireRejected(nullptr, invalid);
	requireRejected("0", "0,1");
	configure(nullptr, "1,0");
	require(CClient().configuredExplorerPlayers() == std::set{PlayerColor(0), PlayerColor(1)}, "plural routing mismatch");
	require(!CClient().configuredExplorerPlayer(), "plural mode selected an arbitrary singular player");

	// Neither side can stand in for the other; cover both real packet orders.
	for(int first = 0; first < 2; ++first)
	{
		CClient client;
		logger.terminalMessages = 0;
		require(!client.recordExplorerTerminal(PlayerColor(2), true), "unconfigured side completed self-play");
		require(!client.recordExplorerTerminal(PlayerColor(first), true), "first result completed self-play early");
		require(!client.recordExplorerTerminal(PlayerColor(first), true), "duplicate first result completed self-play");
		require(logger.terminalMessages == 1, "duplicate or unrelated result was logged as a terminal");
		require(client.recordExplorerTerminal(PlayerColor(1 - first), false), "both actual results failed to complete self-play");
		require(!client.recordExplorerTerminal(PlayerColor(first), true), "completed self-play repeated shutdown");
		require(logger.terminalMessages == 2, "expected exactly two distinct terminals");
	}
	contextualMode = true;
	configure(nullptr, "0,1");
	CClient contextualClient;
	require(!contextualClient.recordExplorerTerminal(PlayerColor(0), false), "contextual first terminal completed early");
	bool duplicateRejected = false;
	try { contextualClient.recordExplorerTerminal(PlayerColor(0), false); }
	catch(const std::runtime_error &) { duplicateRejected = true; }
	require(duplicateRejected, "contextual duplicate terminal was silently accepted");
	require(contextualClient.recordExplorerTerminal(PlayerColor(1), true), "contextual pair did not complete");
	contextualMode = false;
	configure(nullptr, "0,1,2");
	CClient threePlayers;
	require(!threePlayers.recordExplorerTerminal(PlayerColor(0), true), "first of three completed early");
	require(!threePlayers.recordExplorerTerminal(PlayerColor(2), false), "second of three completed early");
	require(threePlayers.recordExplorerTerminal(PlayerColor(1), false), "third terminal did not complete");
	// Compile and exercise the exact mixed-controller selector too. Neither
	// native terminal can stand in for the other or create a second Explorer.
	contextualMode = true;
	setenv("EXPLORER_RL_BUILTIN_OPPONENT", "Nullkiller", 1);
	setenv("EXPLORER_RL_INTERFACE", "adventure-dialogs-v1", 1);
	for(int learner = 0; learner < 2; ++learner)
	{
		const auto value = std::to_string(learner);
		configure(nullptr, value.c_str());
		require(experimentBuiltinLearner() == learner, "mixed controller selected another learner");
		require(CClient().configuredExplorerPlayers() == std::set{PlayerColor(learner)}, "native opponent became Explorer");
		for(int first = 0; first < 2; ++first)
		{
			CClient client;
			logger.terminalMessages = 0;
			bool rejected = false;
			try { client.recordBuiltinNativeTerminal(PlayerColor(2), true); }
			catch(const std::runtime_error &) { rejected = true; }
			require(rejected, "unrelated native terminal was accepted");
			require(!client.recordBuiltinNativeTerminal(PlayerColor(first), true), "one native callback completed mixed game early");
			rejected = false;
			try { client.recordBuiltinNativeTerminal(PlayerColor(first), true); }
			catch(const std::runtime_error &) { rejected = true; }
			require(rejected, "repeated native terminal was accepted");
			require(client.recordBuiltinNativeTerminal(PlayerColor(1 - first), false), "both native callbacks did not complete mixed game");
			require(logger.terminalMessages == 3, "expected two native logs and exactly one Explorer terminal");
		}
	}
	for(const auto invalid : {"", "0,1", "2", "00"})
	{
		configure(nullptr, invalid);
		bool rejected = false;
		try { experimentBuiltinLearner(); }
		catch(const std::runtime_error &) { rejected = true; }
		require(rejected, "ambiguous mixed controller configuration was accepted");
	}
	configure("0", nullptr);
	bool singularRejected = false;
	try { experimentBuiltinLearner(); }
	catch(const std::runtime_error &) { singularRejected = true; }
	require(singularRejected, "legacy singular flag silently enabled mixed evaluation");
	configure(nullptr, "0");
	for(const auto invalid : {"", "Nullkiller2", "ExplorerRL"})
	{
		setenv("EXPLORER_RL_BUILTIN_OPPONENT", invalid, 1);
		bool rejected = false;
		try { experimentBuiltinLearner(); }
		catch(const std::runtime_error &) { rejected = true; }
		require(rejected, "unapproved native opponent was accepted");
	}
	unsetenv("EXPLORER_RL_BUILTIN_OPPONENT");
	unsetenv("EXPLORER_RL_INTERFACE");
	configure(nullptr, nullptr);
	std::cout << "Self-play routing regression passed (strict indices, both terminal orders and mixed controllers).\n";
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
