#pragma once

#include "ArmyOperationRules.h"

namespace explorer_rl
{
/// Rights displayed by an actual native garrison query. A locked army may
/// receive units and rearrange its own slots, but cannot give units away.
struct ArmyAccess
{
	bool owned;
	bool unflaggable;
	bool removable;
};

inline const char * objectGarrisonRole(bool allied, bool removable)
{
	return allied ? (removable ? "allied_garrison" : "allied_locked_garrison")
		: (removable ? "object_garrison" : "locked_garrison");
}

/// Check native request rights independently of the arithmetic/last-stack
/// audit. An allied row is visible, but its units never become ours to move.
inline void validateArmyOperationAccess(ArmyOperation operation, bool sameArmy,
	int64_t sourceCount, int64_t destinationCount, ArmyAccess source, ArmyAccess destination)
{
	const auto canRemove = [sameArmy](ArmyAccess army)
	{
		return (army.owned || army.unflaggable) && (sameArmy || army.removable);
	};
	if(operation == ArmyOperation::SPLIT)
	{
		// Native split is stricter than merge/swap for unflaggable neutrals.
		if(!source.owned || (!sameArmy && !source.removable))
			throw std::invalid_argument("Split cannot remove units from this displayed army");
	}
	else if(operation == ArmyOperation::MERGE)
	{
		if(!canRemove(source))
			throw std::invalid_argument("Merge cannot remove units from this displayed army");
	}
	else if((sourceCount > 0 && !canRemove(source))
		|| (destinationCount > 0 && !canRemove(destination)))
		throw std::invalid_argument("Swap cannot remove units from this displayed army");
}
}
