#include "ArmyOperationRules.h"

#include <cassert>
#include <iostream>
#include <map>

using namespace explorer_rl;

template<class F> void rejected(F operation)
{
	bool failed = false;
	try { operation(); } catch(const std::exception &) { failed = true; }
	assert(failed);
}

std::map<int, int64_t> totals(const TransferArmy & first, const TransferArmy & second)
{
	std::map<int, int64_t> result;
	for(const auto * army : {&first, &second})
		for(const auto & stack : *army) if(stack.count) result[stack.creature] += stack.count;
	return result;
}

int main()
{
	TransferArmy hero{}, town{};
	hero[0] = {0, 25};
	town[3] = {0, 14};
	auto moved = armyOperationEffect(ArmyOperation::SPLIT, town, hero, 3, 0, 6, false, false, true);
	assert((moved.source[3] == TransferStack{0, 8}));
	assert((moved.destination[0] == TransferStack{0, 31}));
	assert(moved.destinationTotal == 31); // the native packet carries the final total
	assert(totals(moved.source, moved.destination) == totals(town, hero));
	auto reverse = armyOperationEffect(ArmyOperation::SPLIT, moved.destination, moved.source, 0, 3, 6, false, true, false);
	assert(reverse.source == hero && reverse.destination == town);
	assert(hero[0].count == 25 && town[3].count == 14); // source snapshots are immutable

	// Same-army full moves and merges preserve the hero's army. The minimum
	// army constraint applies only to removal across different armies.
	auto relocated = armyOperationEffect(ArmyOperation::SPLIT, hero, hero, 0, 6, 25, true, true, true);
	assert(relocated.source == relocated.destination && relocated.source[0].count == 0);
	assert(relocated.source[6] == hero[0]);
	auto divided = armyOperationEffect(ArmyOperation::SPLIT, hero, hero, 0, 5, 12, true, true, true);
	auto merged = armyOperationEffect(ArmyOperation::MERGE, divided.source, divided.source, 5, 0, 0, true, true, true);
	assert(merged.source == hero && merged.destination == hero);
	rejected([&] { armyOperationEffect(ArmyOperation::SPLIT, hero, town, 0, 3, 25, false, true, false); });
	rejected([&] { armyOperationEffect(ArmyOperation::MERGE, hero, town, 0, 3, 0, false, true, false); });
	assert(armyOperationEffect(ArmyOperation::SPLIT, hero, town, 0, 3, 24, false, true, false).source[0].count == 1);

	// A full occupied-to-occupied swap is permitted even for each hero's
	// last stack, and for matching creature IDs. It performs no addition.
	TransferArmy other{};
	other[4] = {1, 7};
	auto swapped = armyOperationEffect(ArmyOperation::SWAP, hero, other, 0, 4, 0, false, true, true);
	assert(swapped.source[0] == other[4] && swapped.destination[4] == hero[0]);
	other[4] = {0, 7};
	auto sameCreature = armyOperationEffect(ArmyOperation::SWAP, hero, other, 0, 4, 0, false, true, true);
	assert(sameCreature.source[0].count == 7 && sameCreature.destination[4].count == 25);
	auto within = armyOperationEffect(ArmyOperation::SWAP, hero, hero, 0, 6, 0, true, true, true);
	assert(within.source == relocated.source);
	rejected([&] { armyOperationEffect(ArmyOperation::SWAP, hero, other, 0, 6, 0, false, true, true); });
	rejected([&] { armyOperationEffect(ArmyOperation::SWAP, other, hero, 6, 0, 0, false, true, true); });
	assert(armyOperationEffect(ArmyOperation::SWAP, town, hero, 3, 6, 0, false, false, true).source[3].count == 0);

	for(const auto operation : {ArmyOperation::SPLIT, ArmyOperation::MERGE, ArmyOperation::SWAP})
	{
		const int count = operation == ArmyOperation::SPLIT ? 1 : 0;
		rejected([&] { armyOperationEffect(operation, hero, hero, 0, 0, count, true, true, true); });
		for(int invalid : {-1, 7})
		{
			rejected([&] { armyOperationEffect(operation, town, hero, invalid, 0, count, false, false, true); });
			rejected([&] { armyOperationEffect(operation, town, hero, 3, invalid, count, false, false, true); });
		}
	}
	for(int count : {-1, 0, 26})
		rejected([&] { armyOperationEffect(ArmyOperation::SPLIT, hero, town, 0, 3, count, false, true, false); });
	rejected([&] { armyOperationEffect(ArmyOperation::SWAP, hero, town, 1, 1, 0, false, true, false); });
	rejected([&] { armyOperationEffect(ArmyOperation::MERGE, hero, town, 0, 1, 0, false, true, false); });
	town[3].creature = 2;
	rejected([&] { armyOperationEffect(ArmyOperation::SPLIT, town, hero, 3, 0, 1, false, false, true); });
	town[3].creature = 0;
	hero[0].count = std::numeric_limits<int32_t>::max();
	rejected([&] { armyOperationEffect(ArmyOperation::SPLIT, town, hero, 3, 0, 1, false, false, true); });
	assert(armyOperationEffect(ArmyOperation::SWAP, town, hero, 3, 0, 0, false, false, true).source[3].count == hero[0].count);
	hero[0].count = 25;
	TransferArmy bad = town;
	bad[1] = {0, 0};
	rejected([&] { armyOperationEffect(ArmyOperation::SWAP, bad, hero, 3, 0, 0, false, false, true); });
	rejected([&] { armyOperationEffect(ArmyOperation::SPLIT, town, hero, 3, 0, 1, true, false, true); });

	// Native garrison consolidation first prefers matching types, then the
	// first empty slot, then an ordered merge with the town source slot as
	// the preferred merge-from slot. Unrelated slots remain exact.
	auto garrison = garrisonArmyEffect(town, hero);
	assert(garrison.source == TransferArmy{});
	assert((garrison.destination[0] == TransferStack{0, 39}));
	town[3] = {8, 4};
	for(int i = 0; i < 7; ++i) hero[i] = {i, i + 1};
	hero[3] = {1, 9}; // preferred duplicate merges slot3 into slot1
	auto packed = garrisonArmyEffect(town, hero);
	assert((packed.destination[1] == TransferStack{1, 11}));
	assert((packed.destination[3] == TransferStack{8, 4}));
	assert(packed.source == TransferArmy{});
	assert(totals(town, hero) == totals(packed.source, packed.destination));
	hero[3] = {3, 9};
	rejected([&] { garrisonArmyEffect(town, hero); }); // eight distinct creatures
	hero[2] = {1, 5}; // no preferred match, native first duplicate pair1→2
	packed = garrisonArmyEffect(town, hero);
	assert((packed.destination[2] == TransferStack{1, 7}));
	assert((packed.destination[1] == TransferStack{8, 4}));

	std::array<int64_t, 7> prices{1, 0, 2, 0, 0, 0, 10};
	const auto cost = upgradeCost(prices, 7);
	assert((cost == std::array<int64_t, 7>{7, 0, 14, 0, 0, 0, 70}));
	prices.fill(0);
	assert(upgradeCost(prices, std::numeric_limits<int32_t>::max()) == prices);
	prices[6] = std::numeric_limits<int32_t>::max();
	assert(upgradeCost(prices, 1)[6] == prices[6]);
	rejected([&] { upgradeCost(prices, 2); });
	prices[6] = -1;
	rejected([&] { upgradeCost(prices, 1); });
	prices[6] = 1;
	for(int count : {-1, 0}) rejected([&] { upgradeCost(prices, count); });
	std::cout << "explicit army operations, garrison effects and upgrade costs passed\n";
}
