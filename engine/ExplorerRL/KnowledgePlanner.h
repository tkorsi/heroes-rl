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
/// One native adjacent move. Transit follows the native pathfinder layer:
/// flying/water-walking steps bypass visits, and landing resumes normal visits.
/// This command metadata is internal to execution, never a policy input.
struct RouteStep
{
	int3 position;
	bool transit = false;
};

/// A route the player's own knowledge supports. Only the current-turn prefix is
/// kept: the contextual interface never executes, reports or exposes the rest.
struct RoutePlan
{
	bool known = false; ///< a route exists over known tiles in the supported travel modes
	std::vector<RouteStep> steps; ///< current-turn moves, origin excluded, ending on land
	bool complete = false; ///< the prefix ends at the destination
};

/// Plans an owned hero's land route, including active flight/water walking,
/// to a destination visitable tile
/// with VCMI's own pathfinder, fed only what the player's callback reveals:
/// tiles its team has explored, objects it can see on them, its own hero and
/// public rules. Unknown tiles are impassable rock, guards come from visible
/// monsters only, and towns and garrisons are judged by ownership, never by
/// their hidden garrisons. Native movement bonuses, costs, layer transitions
/// and same-turn landing rules apply. Boats, teleports and implicit spell casts
/// are outside the travel modes. The caller must hold the game-state read
/// lock, and handles the hero's own tile and an empty movement pool first.
RoutePlan planRoute(const CCallback & player, const CGHeroInstance & hero, const int3 & destination);
}
