#pragma once

#include "../lib/json/JsonNode.h"

VCMI_LIB_NAMESPACE_BEGIN
class CGameState;
struct CPackForClient;
struct CPackForServer;
VCMI_LIB_NAMESPACE_END

/// A second pass over independently regenerated server packets. Its only
/// recorded execution inputs are policy responses, automatic battle inputs,
/// explicitly bound native-opponent requests in mixed mode, and the order of
/// atomic client/adapter transactions. Learner observations remain outputs.
class ContextualReplay
{
	std::vector<std::byte> initialState;
	std::vector<std::vector<std::byte>> packets;
	std::vector<JsonNode> inputs;
	JsonNode terminalInput;
public:
	/// Select explicit controller roles before replaying any server inputs.
	static void configureControllers(const JsonNode & header);
	void initialize(CGameState & state);
	void packet(const CPackForClient & pack);
	void input(const JsonNode & record);
	/// Called only after final natural input effects, state and barrier verify.
	void verifiedTerminalInput(const CPackForServer & request, int64_t sequence,
		uint64_t firstClientPacket, uint64_t lastClientPacket);
	JsonNode run(const std::vector<JsonNode> & tape, bool prefix);
};
