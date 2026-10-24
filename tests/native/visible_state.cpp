// Exercises the production ExplorerRL visibility tracker with synthetic worlds.
// The fixture owns hidden state; the tracker may only learn what tileVisible
// and visibleObjectsAt expose, exactly like the engine adapter.
#include "VisibleState.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>

using explorer_rl::VisibleChange;
using explorer_rl::VisibleObject;
using explorer_rl::VisibleTracker;

namespace
{
[[noreturn]] void fail(const char * message)
{
	std::cerr << "FAILED: " << message << '\n';
	std::exit(1);
}

void expect(bool condition, const char * message)
{
	if(!condition)
		fail(message);
}

struct World
{
	int sizeX = 6;
	int sizeY = 5;
	std::set<std::array<int, 3>> visible; // one team's fog: tiles it currently sees
	std::vector<VisibleObject> objects; // complete hidden truth
	mutable int hiddenQueries = 0;

	int width() const { return sizeX; }
	int height() const { return sizeY; }
	int levels() const { return 1; }
	bool tileVisible(int x, int y, int z) const { return visible.contains({x, y, z}); }

	void visibleObjectsAt(int x, int y, int z, std::vector<VisibleObject> & out) const
	{
		if(!tileVisible(x, y, z))
			++hiddenQueries; // the tracker must never ask about a hidden tile
		for(const auto & object : objects)
			if(object.x == x && object.y == y && object.z == z)
				out.push_back(object);
	}

	VisibleObject * find(int64_t id)
	{
		for(auto & object : objects)
			if(object.id == id)
				return &object;
		return nullptr;
	}

	void remove(int64_t id)
	{
		std::erase_if(objects, [id](const auto & object) { return object.id == id; });
	}

	void reveal(int x0, int x1)
	{
		for(int x = x0; x <= x1; ++x)
			for(int y = 0; y < sizeY; ++y)
				visible.insert({x, y, 0});
	}
};

VisibleChange capture(VisibleTracker & tracker, const World & world)
{
	const auto change = tracker.update(explorer_rl::captureVisible(world));
	expect(world.hiddenQueries == 0, "objects were requested for a hidden tile");
	return change;
}

bool mentionsTile(const VisibleChange & change, int x, int y)
{
	auto at = [x, y](const VisibleObject & object) { return object.x == x && object.y == y; };
	return std::any_of(change.appeared.begin(), change.appeared.end(), at)
		|| std::any_of(change.changed.begin(), change.changed.end(), at);
}

World twoSidedWorld()
{
	World world;
	world.reveal(0, 2); // columns 0..2 visible, 3..5 hidden
	world.objects = {
		{1, 1, 1, 0, 34, 0, 0},   // own hero
		{2, 2, 3, 0, 34, 1, 1},   // enemy hero, visible
		{3, 4, 2, 0, 34, 2, 1},   // enemy hero, hidden
		{4, 1, 4, 0, 53, 0, 255}, // neutral mine, visible
	};
	return world;
}

void baselineAndReconstruction()
{
	World world = twoSidedWorld();
	VisibleTracker tracker;
	const auto first = capture(tracker, world);
	expect(first.appeared.size() == 3, "baseline must report each visible object once");
	expect(first.revealed.size() == 15 && first.hidden.empty(), "baseline must report the visible tiles");
	for(const auto & object : first.appeared)
		expect(object.id != 3, "baseline disclosed a hidden enemy");
	expect(capture(tracker, world).empty(), "an unchanged view must not produce an event");
}

void visibleToHiddenMove()
{
	World world = twoSidedWorld();
	VisibleTracker tracker;
	capture(tracker, world);
	world.find(2)->x = 4; // the visible enemy walks into fog
	const auto change = capture(tracker, world);
	expect(change.disappeared == std::vector<int64_t>{2}, "leaving view must be a bare disappearance");
	expect(change.appeared.empty() && change.changed.empty(), "hidden destination leaked as a fact");
	expect(!mentionsTile(change, 4, 3), "hidden endpoint leaked");
}

void transientSightingBetweenDecisions()
{
	World world = twoSidedWorld();
	VisibleTracker tracker;
	capture(tracker, world); // our decision snapshot
	world.find(3)->x = 2;    // the hidden enemy crosses into view during the opponent's turn
	const auto seen = capture(tracker, world);
	expect(seen.appeared.size() == 1 && seen.appeared[0].id == 3 && seen.appeared[0].x == 2 && seen.appeared[0].y == 2,
		"transient sighting must be reported with its visible position");
	world.find(3)->x = 5;    // and leaves again before our next decision
	const auto gone = capture(tracker, world);
	expect(gone.disappeared == std::vector<int64_t>{3} && !mentionsTile(gone, 5, 2), "departure leaked the hidden tile");
	expect(capture(tracker, world).empty(), "next decision snapshot has nothing new, so the event stream is required");
}

void hiddenAndVisibleRemoval()
{
	World world = twoSidedWorld();
	VisibleTracker tracker;
	capture(tracker, world);
	world.remove(3); // hidden enemy dies out of sight
	expect(capture(tracker, world).empty(), "hidden removal must not produce any event");
	world.remove(2); // visible enemy removed in view
	const auto change = capture(tracker, world);
	expect(change.disappeared == std::vector<int64_t>{2} && change.appeared.empty() && change.changed.empty(),
		"visible removal is only a disappearance");
}

void visibleOwnerChangeAndRehiding()
{
	World world = twoSidedWorld();
	VisibleTracker tracker;
	capture(tracker, world);
	world.find(4)->owner = 1;
	const auto flagged = capture(tracker, world);
	expect(flagged.changed.size() == 1 && flagged.changed[0].owner == 1, "visible owner change must be reported");
	for(int y = 0; y < world.sizeY; ++y)
		world.visible.erase({1, y, 0}); // a re-hiding effect covers column 1
	const auto covered = capture(tracker, world);
	expect(covered.hidden.size() == 5, "re-hidden tiles must be reported");
	expect(covered.disappeared == std::vector<int64_t>({1, 4}), "objects under re-hidden tiles leave the view");
}

void independentPlayers()
{
	World red = twoSidedWorld();
	World blue = twoSidedWorld();
	blue.visible.clear();
	blue.reveal(3, 5); // Blue sees the other half
	VisibleTracker redTracker;
	VisibleTracker blueTracker;
	capture(redTracker, red);
	const auto blueFirst = capture(blueTracker, blue);
	expect(blueFirst.appeared.size() == 1 && blueFirst.appeared[0].id == 3, "Blue must only see its own visible half");
	red.find(3)->y = 3; // the same hidden-to-Red move, visible to Blue
	blue.find(3)->y = 3;
	expect(capture(redTracker, red).empty(), "Red learned about a move only Blue can see");
	const auto blueMove = capture(blueTracker, blue);
	expect(blueMove.changed.size() == 1 && blueMove.changed[0].y == 3, "Blue must see the move in its own view");
}

void randomizedReconstruction()
{
	std::mt19937 rng(12345);
	for(int trial = 0; trial < 200; ++trial)
	{
		World world;
		for(int64_t id = 1; id <= 8; ++id)
			world.objects.push_back({id, static_cast<int>(rng() % world.sizeX), static_cast<int>(rng() % world.sizeY), 0, 34, 0, static_cast<int>(rng() % 3)});
		VisibleTracker tracker;
		std::map<int64_t, VisibleObject> folded;
		std::set<std::array<int, 3>> foldedTiles;
		for(int step = 0; step < 40; ++step)
		{
			const int operation = static_cast<int>(rng() % 4);
			if(operation == 0)
				world.visible.insert({static_cast<int>(rng() % world.sizeX), static_cast<int>(rng() % world.sizeY), 0});
			else if(operation == 1 && !world.visible.empty())
				world.visible.erase(std::next(world.visible.begin(), static_cast<long>(rng() % world.visible.size())));
			else if(operation == 2 && !world.objects.empty())
			{
				auto & object = world.objects[rng() % world.objects.size()];
				object.x = std::clamp(object.x + static_cast<int>(rng() % 3) - 1, 0, world.sizeX - 1);
				object.y = std::clamp(object.y + static_cast<int>(rng() % 3) - 1, 0, world.sizeY - 1);
			}
			else if(!world.objects.empty())
				world.objects.erase(world.objects.begin() + static_cast<long>(rng() % world.objects.size()));
			const auto change = capture(tracker, world);
			for(const auto & object : change.appeared)
				folded[object.id] = object;
			for(const auto & object : change.changed)
				folded[object.id] = object;
			for(const auto id : change.disappeared)
				folded.erase(id);
			for(const auto & tile : change.revealed)
				foldedTiles.insert(tile);
			for(const auto & tile : change.hidden)
				foldedTiles.erase(tile);
			expect(foldedTiles == world.visible, "folded tile changes differ from the team's fog");
			std::map<int64_t, VisibleObject> truth;
			for(const auto & object : world.objects)
				if(world.visible.contains({object.x, object.y, object.z}))
					truth.emplace(object.id, object);
			expect(folded == truth, "folded object changes differ from the visible objects");
		}
	}
}
}

int main()
{
	baselineAndReconstruction();
	visibleToHiddenMove();
	transientSightingBetweenDecisions();
	hiddenAndVisibleRemoval();
	visibleOwnerChangeAndRehiding();
	independentPlayers();
	randomizedReconstruction();
	std::cout << "visible-state tracker: 7 scenarios, 200 randomized folds\n";
	return 0;
}
