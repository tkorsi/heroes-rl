#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace explorer_rl
{
struct TransferStack
{
	int creature = -1;
	int64_t count = 0;
	bool operator==(const TransferStack &) const = default;
};
using TransferArmy = std::array<TransferStack, 7>;

struct TransferEffect
{
	TransferArmy source;
	TransferArmy destination;
	int32_t destinationTotal;
};

/// Exact forward transfer expressed in the native split request's FINAL
/// destination count. No preferred slot, clamping, swap, or implicit leave-one.
inline TransferEffect transferEffect(const TransferArmy & source, const TransferArmy & destination,
	int sourceSlot, int destinationSlot, int64_t count, bool sourceRequiresNonempty)
{
	constexpr auto maximum = std::numeric_limits<int32_t>::max();
	const auto valid = [](const TransferArmy & army)
	{
		for(const auto & stack : army)
			if(stack.count < 0 || stack.count > maximum || (stack.count == 0 ? stack.creature != -1 : stack.creature < 0))
				throw std::invalid_argument("Unsupported transfer army stack");
	};
	valid(source);
	valid(destination);
	if(sourceSlot < 0 || sourceSlot >= 7 || destinationSlot < 0 || destinationSlot >= 7 || count <= 0)
		throw std::invalid_argument("Invalid explicit transfer slot or count");
	const auto & from = source[sourceSlot];
	const auto & to = destination[destinationSlot];
	if(from.count < count || (to.count && to.creature != from.creature))
		throw std::invalid_argument("Transfer exceeds source or would swap unlike creatures");
	// ArrangeStacks and the native split UI both calculate the combined count
	// in int32, even when the selected transfer is smaller than that total.
	if(from.count + to.count > maximum)
		throw std::overflow_error("Unsupported transfer combined stack count");
	int nonempty = 0;
	for(const auto & stack : source) nonempty += stack.count != 0;
	if(sourceRequiresNonempty && nonempty == 1 && count == from.count)
		throw std::invalid_argument("Transfer would remove a hero's final creature");
	TransferEffect result{source, destination, static_cast<int32_t>(to.count + count)};
	result.source[sourceSlot].count -= count;
	if(result.source[sourceSlot].count == 0) result.source[sourceSlot].creature = -1;
	result.destination[destinationSlot] = {from.creature, to.count + count};
	return result;
}
}
