// Actual request-accounting implementation, real JsonNode and native packet
// serialization. No game, engine process, learner or captured episode is run.
#include "StdInc.h"
#include "RequestAccounting.h"
#include "../lib/serializer/BinarySerializer.h"
#include "../lib/serializer/BinaryDeserializer.h"

#include <iostream>

// PRODUCTION_PACKET_CODEC
// PRODUCTION_AUTOMATIC_ALLOWED

namespace
{
int checks = 0;
void check(bool value)
{
	if(!value) throw std::runtime_error("terminal request accounting assertion failed");
	++checks;
}

struct Fixture
{
	std::vector<JsonNode> tape;
	std::vector<JsonNode> inputs;
	JsonNode terminal;
	bool prefix = false;
	int learner = 0;
	int tailKind = 0;

	JsonNode request(const CPackForServer & pack, const char * source) const
	{
		JsonNode value;
		value["kind"].String() = "request";
		value["seq"].Integer() = 7;
		value["pack"].String() = toHex(encodePack(pack));
		value["player"].Integer() = pack.player.getNum();
		value["request_id"].Integer() = pack.requestID;
		value["pack_type"].Integer() = CTypeList::getInstance().getTypeID(&pack);
		value["source"].String() = source;
		return value;
	}
	void add(const CPackForServer & pack, const char * source, const char * operation, bool applied)
	{
		JsonNode enter;
		enter["kind"].String() = "enter";
		enter["actor"].String() = "client";
		enter["operation"].String() = operation;
		enter["packet"]["index"].Integer() = 17;
		enter["packet"]["completed"].Integer() = 17;
		enter["packet"]["phase"].String() = "callbacks_complete";
		enter["packet"]["type"].Integer() = 117;
		tape.push_back(enter);
		const auto value = request(pack, source);
		tape.push_back(value);
		JsonNode leave;
		leave["kind"].String() = "leave";
		leave["actor"].String() = "client";
		leave["operation"].String() = operation;
		tape.push_back(leave);
		if(applied)
		{
			JsonNode input;
			input["pack"] = value["pack"];
			input["type"] = value["pack_type"];
			input["player"] = value["player"];
			input["request"] = value["request_id"];
			inputs.push_back(input);
		}
	}

	JsonNode tailRequest(int player, uint32_t id) const
	{
		if(tailKind == 1)
		{
			BuildStructure build(ObjectInstanceID(0), BuildingID::TAVERN);
			build.player = PlayerColor(player);
			build.requestID = id;
			return request(build, "native_opponent");
		}
		if(tailKind == 2)
		{
			EndTurn end;
			end.player = PlayerColor(player); end.requestID = id;
			return request(end, "native_opponent");
		}
		if(tailKind == 3)
		{
			TradeOnMarketplace trade;
			trade.player = PlayerColor(player); trade.requestID = id;
			trade.marketId = ObjectInstanceID(0); trade.heroId = ObjectInstanceID(3);
			trade.mode = EMarketMode::RESOURCE_RESOURCE;
			trade.r1 = {TradeItemSell(GameResID(0))};
			trade.r2 = {TradeItemBuy(GameResID(6))};
			trade.val = {1};
			return request(trade, "native_opponent");
		}
		MoveHero move;
		move.player = PlayerColor(player);
		move.requestID = id;
		move.hid = ObjectInstanceID(3);
		move.path = {int3(8, 7, 0)};
		return request(move, "native_opponent");
	}

	explicit Fixture(int side = 0, int kind = 0) : learner(side), tailKind(kind)
	{
		AdvInterfaceReady ready;
		ready.player = PlayerColor(side);
		ready.requestID = 10;
		add(ready, "automatic", "request", true);
		MakeAction battle;
		battle.player = PlayerColor(side);
		battle.requestID = 11;
		add(battle, "automatic", "packet", true);
		MoveHero move;
		move.player = PlayerColor(1 - side);
		move.requestID = 12;
		move.hid = ObjectInstanceID(3);
		move.path = {int3(8, 7, 0)};
		add(move, "native_opponent", "request", false);
		tape[7] = tailRequest(1 - side, 12);
		JsonNode close;
		close["kind"].String() = "close";
		close["barrier_id"].Integer() = 1;
		close["client_packets"].Integer() = 20;
		tape.push_back(close);
		terminal["input_seq"].Integer() = 1;
		terminal["request_id"].Integer() = 11;
		terminal["player"].Integer() = side;
		terminal["pack_type"] = inputs.back()["type"];
		terminal["first_client_packet"].Integer() = 11;
		terminal["last_client_packet"].Integer() = 20;
	}
	JsonNode run() const
	{
		return contextualRequestAccounting(tape, inputs, terminal, prefix, learner,
			[](const std::string & encoded) { return decodePack(fromHex(encoded), nullptr); }, automaticAllowed);
	}
};

void rejected(const std::function<void(Fixture &)> & mutate)
{
	for(const int side : {0, 1})
		for(const int tailKind : {0, 1, 2, 3})
		{
			Fixture fixture(side, tailKind);
			mutate(fixture);
			try { fixture.run(); }
			catch(const std::exception &) { ++checks; continue; }
			throw std::runtime_error("invalid terminal request was admitted");
		}
}
}

int main()
{
	try
	{
		for(const int side : {0, 1})
		for(const int tailKind : {0, 1, 2, 3})
		{
			Fixture f(side, tailKind);
			const auto actual = f.run();
			check(actual["authoritative_inputs"].Integer() == 2);
			check(actual["terminal_unapplied"].Vector().size() == 1);
			check(actual["terminal_unapplied"][0]["pack"] == f.tape[7]["pack"]);
			check(actual["terminal_unapplied"][0]["receipt"] == f.tape[6]["packet"]);
			check(actual["final_terminal_input"] == f.terminal);
			check(actual["terminal_unapplied"][0]["pack_type"].Integer() == (tailKind == 3 ? 193 : tailKind == 2 ? 180 : tailKind == 1 ? 185 : 182));
			for(const int position : {11, 19})
			{
				f.tape[6]["packet"]["completed"].Integer() = position;
				f.tape[6]["packet"]["index"].Integer() = position;
				check(f.run()["terminal_unapplied"].Vector().size() == 1);
			}
			f.tape.erase(f.tape.begin() + 6, f.tape.begin() + 9);
			check(f.run()["terminal_unapplied"].Vector().empty());
			f.prefix = true; f.terminal = JsonNode(); f.tape.pop_back();
			check(f.run()["terminal_unapplied"].Vector().empty());
			f.learner = -1;
			check(f.run()["terminal_unapplied"].Vector().empty());
		}
		rejected([](auto & f) { f.prefix = true; });
		rejected([](auto & f) { f.learner = -1; });
		rejected([](auto & f) { f.terminal = JsonNode(); });
		rejected([](auto & f) { f.terminal["input_seq"].Integer() = 0; });
		rejected([](auto & f) { f.terminal["request_id"].Integer() = 10; });
		rejected([](auto & f) { f.terminal["player"].Integer() = 1 - f.learner; });
		rejected([](auto & f) { f.terminal["pack_type"].Integer() = 182; });
		rejected([](auto & f) { f.tape[7]["source"].String() = "adapter"; });
		rejected([](auto & f) { f.tape[7]["source"].String() = "automatic"; });
		rejected([](auto & f) { f.tape[6]["actor"].String() = "player1"; });
		rejected([](auto & f) { f.tape[6]["operation"].String() = "packet"; });
		rejected([](auto & f) { f.tape[7]["player"].Integer() = f.learner; });
		rejected([](auto & f) { f.tape[7]["request_id"].Integer() = 13; });
		rejected([](auto & f) { f.tape[7]["request_id"].Float() = 12.0; });
		rejected([](auto & f) { f.tape[7]["seq"].Float() = 7.0; });
		rejected([](auto & f) { f.tape[7]["pack_type"].Integer() = 198; });
		rejected([](auto & f) { f.tape[7]["pack_type"].Integer() = f.tailKind == 1 ? 182 : 185; });
		rejected([](auto & f) { f.tape[7]["pack"].String() += "00"; });
		rejected([](auto & f) { f.tape[7]["pack"].String() = "not-hex"; });
		for(const auto id : {11u, 13u, 0u})
			rejected([id](auto & f) { f.tape[7] = f.tailRequest(1 - f.learner, id); });
		rejected([](auto & f) { f.tape[7] = f.tailRequest(f.learner, 12); });
		rejected([](auto & f) { DismissHero other; other.player = PlayerColor(1 - f.learner); other.requestID = 12;
			f.tape[7] = f.request(other, "native_opponent"); });
		rejected([](auto & f) { const std::vector<JsonNode> extra(f.tape.begin() + 6, f.tape.begin() + 9);
			f.tape.insert(f.tape.begin() + 9, extra.begin(), extra.end()); });
		rejected([](auto & f) {
			MakeAction battle; battle.player = PlayerColor(f.learner); battle.requestID = std::numeric_limits<uint32_t>::max();
			f.tape[4] = f.request(battle, "automatic");
			f.inputs.back()["pack"] = f.tape[4]["pack"];
			f.inputs.back()["request"] = f.tape[4]["request_id"];
			f.terminal["request_id"] = f.tape[4]["request_id"];
			f.tape[7] = f.tailRequest(1 - f.learner, 0);
		});
		rejected([](auto & f) { f.tape.pop_back(); });
		rejected([](auto & f) { f.tape.push_back(f.tape[7]); });
		rejected([](auto & f) { JsonNode response; response["kind"].String() = "response";
			response["player"].Integer() = f.learner; f.tape.insert(f.tape.begin() + 8, response); });
		rejected([](auto & f) { f.tape.back()["client_packets"].Integer() = 21; });
		rejected([](auto & f) { f.tape.back()["barrier_id"].Integer() = 2; });
		rejected([](auto & f) { f.tape[6]["packet"]["completed"].Integer() = 16; });
		rejected([](auto & f) { f.tape[6]["packet"]["phase"].String() = "state_applied"; });
		rejected([](auto & f) { f.tape[6]["packet"]["type"].Bool() = true; });
		for(const int position : {0, 10, 20, 21})
			rejected([position](auto & f) { f.tape[6]["packet"]["completed"].Integer() = position;
				f.tape[6]["packet"]["index"].Integer() = position; });
		rejected([](auto & f) { f.inputs[0]["pack"] = f.inputs[1]["pack"]; });
		rejected([](auto & f) { f.inputs[0]["type"].Integer() = 182; });
		rejected([](auto & f) { f.inputs[0]["player"].Integer() = 1 - f.learner; });
		rejected([](auto & f) { f.inputs[0]["request"].Integer() = 9; });
		rejected([](auto & f) { f.tape.erase(f.tape.begin(), f.tape.begin() + 3); });
		rejected([](auto & f) { std::swap(f.inputs[0], f.inputs[1]); });
		rejected([](auto & f) { f.inputs.push_back(f.inputs.back()); });
		rejected([](auto & f) { f.inputs.clear(); });
		// Exact retained B-v2 episode 5 tail: decoding this intent does not
		// apply construction, charge resources, or change the original cohort.
		const auto captured = decodePack(fromHex("0000b90101da000005"), nullptr);
		const auto * build = dynamic_cast<const BuildStructure *>(captured.get());
		check(build != nullptr);
		check(build->player == PlayerColor(1) && build->requestID == 90);
		check(build->tid == ObjectInstanceID(0) && build->bid == BuildingID::TAVERN);
		const auto capturedEnd = decodePack(fromHex("0000b40101c201"), nullptr);
		const auto * end = dynamic_cast<const EndTurn *>(capturedEnd.get());
		check(end != nullptr && end->player == PlayerColor(1) && end->requestID == 194);
		std::cout << "native terminal request accounting: " << checks << " checks passed\n";
		return 0;
	}
	catch(const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
