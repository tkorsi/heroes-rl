#pragma once

#include <cstddef>
#include <cstdint>

namespace explorer_rl
{
/// BulkMoveArtifacts removes the source before inserting at the requested
/// backpack index. Same-backpack destinations therefore refer to the final
/// order, not to gaps in the original inventory; its last index is size - 1.
inline bool backpackInsertionFits(std::size_t size, bool removesFromBackpack, int64_t index)
{
	if(index < 0 || (removesFromBackpack && size == 0)) return false;
	return static_cast<uint64_t>(index) <= size - (removesFromBackpack ? 1 : 0);
}
}
