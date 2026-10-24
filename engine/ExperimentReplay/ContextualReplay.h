#pragma once

#include "../lib/json/JsonNode.h"

VCMI_LIB_NAMESPACE_BEGIN
class CGameState;
struct CPackForClient;
VCMI_LIB_NAMESPACE_END

/// A second pass over independently regenerated server packets. Its only
/// recorded execution inputs are policy responses, automatic battle inputs,
/// and the order of atomic client/adapter transactions. Observations are outputs.
class ContextualReplay
{
	std::vector<std::byte> initialState;
	std::vector<std::vector<std::byte>> packets;
	std::vector<JsonNode> inputs;
public:
	void initialize(CGameState & state);
	void packet(const CPackForClient & pack);
	void input(const JsonNode & record);
	JsonNode run(const std::vector<JsonNode> & tape, bool prefix);
};
