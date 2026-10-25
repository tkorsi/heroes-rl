#pragma once

#include "TransferRules.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace explorer_rl
{
enum class ArmyOperation { SPLIT, MERGE, SWAP };

struct ArmyOperationEffect
{
	TransferArmy source;
	TransferArmy destination;
	int32_t destinationTotal = 0; ///< only SPLIT sends a final destination count
};

inline void validateArmy(const TransferArmy & army)
{
	for(const auto & stack : army)
		if(stack.count < 0 || stack.count > std::numeric_limits<int32_t>::max()
			|| (stack.count == 0 ? stack.creature != -1 : stack.creature < 0))
			throw std::invalid_argument("Unsupported army stack");
}

/// Predict one explicitly selected native ArrangeStacks request. SWAP includes
/// occupied matching-creature slots: their counts exchange, never merge. This
/// is an effect audit, never a slot picker or an army-changing implementation.
inline ArmyOperationEffect armyOperationEffect(ArmyOperation operation,
	const TransferArmy & source, const TransferArmy & destination, int sourceSlot,
	int destinationSlot, int64_t count, bool sameArmy, bool sourceRequiresNonempty,
	bool destinationRequiresNonempty)
{
	validateArmy(source);
	validateArmy(destination);
	if(sourceSlot < 0 || sourceSlot >= 7 || destinationSlot < 0 || destinationSlot >= 7
		|| (sameArmy && (source != destination || sourceSlot == destinationSlot)))
		throw std::invalid_argument("Invalid distinct army slots");
	if(operation != ArmyOperation::SPLIT && count != 0)
		throw std::invalid_argument("Only split takes a moved count");
	const auto from = source[sourceSlot];
	const auto to = destination[destinationSlot];
	const auto last = [](const TransferArmy & army)
	{
		return std::count_if(army.begin(), army.end(), [](const auto & stack) { return stack.count != 0; }) == 1;
	};
	ArmyOperationEffect result{source, destination};
	if(operation == ArmyOperation::SWAP)
	{
		if(!from.count && !to.count) throw std::invalid_argument("Cannot swap two empty slots");
		if(!sameArmy && ((from.count && !to.count && sourceRequiresNonempty && last(source))
			|| (to.count && !from.count && destinationRequiresNonempty && last(destination))))
			throw std::invalid_argument("Swap would empty a hero army");
		result.source[sourceSlot] = to;
		if(sameArmy) result.source[destinationSlot] = from;
		else result.destination[destinationSlot] = from;
	}
	else
	{
		if(!from.count || (to.count && to.creature != from.creature)
			|| (operation == ArmyOperation::MERGE && !to.count))
			throw std::invalid_argument("Split or merge requires compatible occupied source");
		if(operation == ArmyOperation::MERGE) count = from.count;
		if(count <= 0 || count > from.count)
			throw std::invalid_argument("Invalid explicit moved count");
		// Native ArrangeStacks computes the combined count in int32 even if
		// the requested move is small. A swap performs no such addition.
		if(from.count + to.count > std::numeric_limits<int32_t>::max())
			throw std::overflow_error("Army combined stack count overflows native int32");
		if(!sameArmy && sourceRequiresNonempty && last(source) && count == from.count)
			throw std::invalid_argument("Move would empty a hero army");
		result.source[sourceSlot] = count == from.count ? TransferStack{} : TransferStack{from.creature, from.count - count};
		const TransferStack finalDestination{from.creature, to.count + count};
		if(sameArmy) result.source[destinationSlot] = finalDestination;
		else result.destination[destinationSlot] = finalDestination;
		result.destinationTotal = static_cast<int32_t>(finalDestination.count);
	}
	if(sameArmy) result.destination = result.source;
	return result;
}

/// Audit the mandatory native moveArmy(town, visitor, true) performed by
/// garrisonSwap. This mirrors getSlotFor and mergeableStacks' ordered choices;
/// it is not exposed as an additional policy action.
inline ArmyOperationEffect garrisonArmyEffect(const TransferArmy & town, const TransferArmy & visitor)
{
	validateArmy(town);
	validateArmy(visitor);
	ArmyOperationEffect result{town, visitor};
	for(int sourceSlot = 0; sourceSlot < 7; ++sourceSlot)
	{
		const auto from = result.source[sourceSlot];
		if(!from.count) continue;
		int target = -1;
		for(int i = 0; i < 7; ++i)
			if(result.destination[i].count && result.destination[i].creature == from.creature) { target = i; break; }
		if(target == -1)
			for(int i = 0; i < 7; ++i)
				if(!result.destination[i].count) { target = i; break; }
		if(target == -1)
		{
			int mergeFrom = -1, mergeTo = -1;
			const auto findPartner = [&](int candidate)
			{
				if(!result.destination[candidate].count) return;
				for(int i = 0; i < 7; ++i)
					if(i != candidate && result.destination[i].count
						&& result.destination[i].creature == result.destination[candidate].creature)
					{ mergeFrom = candidate; mergeTo = i; return; }
			};
			findPartner(sourceSlot); // native preferable slot is the town source slot
			for(int i = 0; i < 7 && mergeFrom == -1; ++i) findPartner(i);
			if(mergeFrom == -1) throw std::invalid_argument("Native garrison has no army capacity");
			const auto merged = armyOperationEffect(ArmyOperation::MERGE, result.destination,
				result.destination, mergeFrom, mergeTo, 0, true, true, true);
			result.destination = merged.destination;
			target = mergeFrom;
		}
		const auto moved = armyOperationEffect(ArmyOperation::SPLIT, result.source, result.destination,
			sourceSlot, target, from.count, false, false, true);
		result.source = moved.source;
		result.destination = moved.destination;
	}
	return result;
}

inline std::array<int64_t, 7> upgradeCost(const std::array<int64_t, 7> & perCreature, int64_t count)
{
	if(count < 1 || count > std::numeric_limits<int32_t>::max())
		throw std::invalid_argument("Invalid upgrade stack count");
	std::array<int64_t, 7> total{};
	for(size_t i = 0; i < total.size(); ++i)
	{
		if(perCreature[i] < 0 || perCreature[i] > std::numeric_limits<int32_t>::max()
			|| perCreature[i] > std::numeric_limits<int32_t>::max() / count)
			throw std::overflow_error("Upgrade cost overflows native resource quantity");
		total[i] = perCreature[i] * count;
	}
	return total;
}
}
