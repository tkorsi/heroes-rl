#include "GarrisonRules.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace explorer_rl;

template<class F> void rejected(F operation)
{
	bool failed = false;
	try { operation(); } catch(const std::invalid_argument &) { failed = true; }
	assert(failed);
}

int main()
{
	const ArmyAccess owned{true, false, true};
	const ArmyAccess locked{true, false, false};
	const ArmyAccess allied{false, false, true};
	const ArmyAccess alliedLocked{false, false, false};
	const ArmyAccess joining{false, true, true};
	assert(std::string(objectGarrisonRole(false, true)) == "object_garrison");
	assert(std::string(objectGarrisonRole(false, false)) == "locked_garrison");
	assert(std::string(objectGarrisonRole(true, true)) == "allied_garrison");
	assert(std::string(objectGarrisonRole(true, false)) == "allied_locked_garrison");

	for(auto destination : {owned, locked, allied, alliedLocked})
	{
		// Gifts remain possible, but occupied swaps take units in both
		// directions and therefore cannot give to a locked/allied stack.
		validateArmyOperationAccess(ArmyOperation::SPLIT, false, 25, 0, owned, destination);
		validateArmyOperationAccess(ArmyOperation::SPLIT, false, 25, 14, owned, destination);
		validateArmyOperationAccess(ArmyOperation::MERGE, false, 25, 14, owned, destination);
		validateArmyOperationAccess(ArmyOperation::SWAP, false, 25, 0, owned, destination);
		validateArmyOperationAccess(ArmyOperation::SWAP, false, 0, 25, destination, owned);
	}
	for(auto restricted : {locked, allied, alliedLocked})
	{
		for(const auto operation : {ArmyOperation::SPLIT, ArmyOperation::MERGE, ArmyOperation::SWAP})
			rejected([&] { validateArmyOperationAccess(operation, false, 25, 14, restricted, owned); });
		rejected([&] { validateArmyOperationAccess(ArmyOperation::SWAP, false, 25, 14, owned, restricted); });
		rejected([&] { validateArmyOperationAccess(ArmyOperation::SWAP, false, 0, 14, owned, restricted); });
	}
	for(const auto operation : {ArmyOperation::SPLIT, ArmyOperation::MERGE, ArmyOperation::SWAP})
	{
		validateArmyOperationAccess(operation, true, 25, 14, locked, locked);
		validateArmyOperationAccess(operation, false, 25, 14, owned, owned);
		for(auto other : {allied, alliedLocked})
			rejected([&] { validateArmyOperationAccess(operation, true, 25, 14, other, other); });
	}
	// Joining neutral armies retain the native distinction: merge/swap
	// work for UNFLAGGABLE, whereas split demands an owned source.
	validateArmyOperationAccess(ArmyOperation::MERGE, false, 14, 25, joining, owned);
	validateArmyOperationAccess(ArmyOperation::SWAP, false, 14, 25, joining, owned);
	rejected([&] { validateArmyOperationAccess(ArmyOperation::SPLIT, false, 14, 25, joining, owned); });

	// Permission never overrides arithmetic or the visiting hero's final
	// stack. Depositing 24 is legal; depositing all 25 is not.
	TransferArmy hero{}, garrison{};
	hero[0] = {0, 25};
	validateArmyOperationAccess(ArmyOperation::SPLIT, false, 25, 0, owned, locked);
	assert(armyOperationEffect(ArmyOperation::SPLIT, hero, garrison, 0, 3, 24, false, true, false).source[0].count == 1);
	rejected([&] { armyOperationEffect(ArmyOperation::SPLIT, hero, garrison, 0, 3, 25, false, true, false); });
	// Exact effect oracle for the native regression fixture: the locked
	// destination's source-index slot holds unrelated archers, while the
	// selected destination slot holds two pikemen. A one-unit gift must
	// conserve all 27 pikemen without inspecting that unrelated slot.
	garrison[0] = {2, 3};
	garrison[1] = {0, 2};
	const auto deposited = armyOperationEffect(ArmyOperation::SPLIT, hero, garrison, 0, 1, 1, false, true, false);
	assert(deposited.source[0].count == 24 && deposited.destination[1].count == 3);
	assert(deposited.destination[0] == garrison[0] && deposited.destinationTotal == 3);
	rejected([&] { validateArmyOperationAccess(ArmyOperation::SPLIT, false, 3, 24, locked, owned); });
	// Locking never prevents an owned army's internal rearrangement.
	validateArmyOperationAccess(ArmyOperation::SPLIT, true, 2, 0, locked, locked);
	const auto internal = armyOperationEffect(ArmyOperation::SPLIT, garrison, garrison, 1, 3, 1, true, false, false);
	assert(internal.source[1].count == 1 && internal.source[3].count == 1);
	assert(internal.source == internal.destination && internal.source[0] == garrison[0]);
	const auto emptyDeposit = armyOperationEffect(ArmyOperation::SPLIT, hero, garrison, 0, 3, 1, false, true, false);
	assert(emptyDeposit.source[0].count == 24 && emptyDeposit.destination[3].count == 1);
	std::cout << "garrison ownership, locking, gifts and last-stack boundaries passed\n";
}
