#pragma once

#include "KnowledgeGrid.h"

#include "../../lib/int3.h"

#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
class CCallback;
class CGHeroInstance;
VCMI_LIB_NAMESPACE_END

namespace explorer_rl
{
/// A route the player's own knowledge supports. Only the current-turn prefix is
/// kept: the contextual interface never executes, reports or exposes the rest.
struct RoutePlan
{
	bool known = false; ///< a route exists over known tiles in the supported travel modes
	std::vector<int3> steps; ///< the current-turn prefix as visitable tiles in walking order, origin excluded
	bool complete = false; ///< the prefix ends at the destination
};

/// Plans an owned hero's ordinary land route to a destination visitable tile
/// with VCMI's own pathfinder, fed only what the player's callback reveals:
/// tiles its team has explored, objects it can see on them, its own hero and
/// public rules. Unknown tiles are impassable rock, guards come from visible
/// monsters only, and towns and garrisons are judged by ownership, never by
/// their hidden garrisons. Boats, flight, water walking, teleports and spells
/// are outside the travel modes. The caller must hold the game-state read
/// lock, and handles the hero's own tile and an empty movement pool first.
RoutePlan planRoute(const CCallback & player, const CGHeroInstance & hero, const int3 & destination);
}
