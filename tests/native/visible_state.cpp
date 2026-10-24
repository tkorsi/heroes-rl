// Exercises the production ExplorerRL visibility tracker with synthetic worlds.
// The fixture owns hidden state; the tracker may only learn what tileVisible,
// tileFacts, visibleObjectsAt and ownState expose, exactly like the engine adapter.
#include "VisibleState.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <string>

using explorer_rl::OwnState;
using explorer_rl::VisibleChange;
using explorer_rl::VisibleObject;
using explorer_rl::VisibleTile;
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
	std::set<std::array<int, 3>> blocking; // tiles with a blocking object, hidden truth too
	OwnState own{std::array<int, 3>{1, 1, 0}, 1500, 1500, {20, 5, 20, 5, 5, 5, 7500}, {{0, 1, 10}}, {}, 0};
	mutable int hiddenQueries = 0;

	int width() const { return sizeX; }
	int height() const { return sizeY; }
	int levels() const { return 1; }
	bool tileVisible(int x, int y, int z) const { return visible.contains({x, y, z}); }
	OwnState ownState() const { return own; }

	VisibleTile tileFacts(int x, int y, int z) const
	{
		if(!tileVisible(x, y, z))
			++hiddenQueries; // the tracker must never ask about a hidden tile
		const bool occupied = std::any_of(objects.begin(), objects.end(),
			[x, y, z](const auto & object) { return object.x == x && object.y == y && object.z == z; });
		return {(x + y) % 10, blocking.contains({x, y, z}), occupied};
	}

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
		{4, 1, 4, 0, 53, 0, -1}, // neutral mine, visible (PlayerColor::NEUTRAL)
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

void tileFactsFollowVisibility()
{
	World world = twoSidedWorld();
	VisibleTracker tracker;
	const auto first = capture(tracker, world);
	for(const auto & fact : first.revealed)
		expect(fact.tile == world.tileFacts(fact.position[0], fact.position[1], fact.position[2]), "revealed tile without its facts");
	world.blocking.insert({4, 4, 0}); // a hidden tile becomes blocked
	world.find(3)->y = 3;             // and the hidden enemy steps to another hidden tile
	expect(capture(tracker, world).empty(), "a hidden tile change produced an event");
	world.find(1)->x = 0;             // our hero steps within view: two tiles change occupancy
	const auto moved = capture(tracker, world);
	expect(moved.tilesChanged.size() == 2 && moved.revealed.empty() && moved.hidden.empty(),
		"visible tile occupancy changes must be reported with current facts");
	for(const auto & fact : moved.tilesChanged)
		expect(fact.position[0] <= 2, "a tile fact was reported for a hidden tile");
	world.reveal(4, 4);               // seeing the column again shows its current facts only
	const auto seen = capture(tracker, world);
	expect(seen.revealed.size() == 5 && seen.tilesChanged.empty(), "re-revealed tiles must arrive as revealed facts");
	for(const auto & fact : seen.revealed)
		if(fact.position == std::array<int, 3>{4, 4, 0})
			expect(fact.tile.blocked, "a revealed tile must carry its current facts");
}

void ownStateChanges()
{
	World world = twoSidedWorld();
	VisibleTracker tracker;
	const auto first = capture(tracker, world);
	expect(first.own && *first.own == world.own, "the first capture must report the complete own state");
	world.own.resources[6] += 1000; // new-day income, no map change
	const auto income = capture(tracker, world);
	expect(income.own && *income.own == world.own && income.appeared.empty() && income.revealed.empty(),
		"an own-state change alone must produce an event with the complete own state");
	expect(!capture(tracker, world).own, "an unchanged own state must not be repeated");
	world.own.hero.reset();
	world.own.army.clear();
	world.own.movement = world.own.movementMax = 0;
	const auto lost = capture(tracker, world);
	expect(lost.own && !lost.own->hero && lost.own->army.empty(), "losing the hero must be reported as an own-state change");
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
		std::map<std::array<int, 3>, VisibleTile> foldedFacts;
		std::optional<OwnState> foldedOwn;
		for(int step = 0; step < 40; ++step)
		{
			const int operation = static_cast<int>(rng() % 6);
			if(operation == 4)
				world.own.resources[rng() % world.own.resources.size()] += static_cast<int>(rng() % 3);
			else if(operation == 5)
			{
				const std::array<int, 3> tile{static_cast<int>(rng() % world.sizeX), static_cast<int>(rng() % world.sizeY), 0};
				if(!world.blocking.erase(tile))
					world.blocking.insert(tile);
			}
			else
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
			for(const auto & fact : change.revealed)
			{
				expect(!foldedTiles.contains(fact.position), "a visible tile was revealed again");
				foldedTiles.insert(fact.position);
				foldedFacts[fact.position] = fact.tile;
			}
			for(const auto & fact : change.tilesChanged)
			{
				expect(foldedTiles.contains(fact.position), "facts changed for a tile the player does not see");
				foldedFacts[fact.position] = fact.tile;
			}
			for(const auto & tile : change.hidden)
			{
				foldedTiles.erase(tile);
				foldedFacts.erase(tile);
			}
			if(change.own)
				foldedOwn = *change.own;
			expect(foldedTiles == world.visible, "folded tile changes differ from the team's fog");
			for(const auto & tile : world.visible)
				expect(foldedFacts.at(tile) == world.tileFacts(tile[0], tile[1], tile[2]), "folded tile facts differ from the visible tiles");
			expect(foldedOwn == world.own, "folded own state differs from the player's own state");
			std::map<int64_t, VisibleObject> truth;
			for(const auto & object : world.objects)
				if(world.visible.contains({object.x, object.y, object.z}))
					truth.emplace(object.id, object);
			expect(folded == truth, "folded object changes differ from the visible objects");
		}
	}
}
}

/// The replay digest covers exactly the player's view: hidden truth never
/// changes it, every visible fact does, and the Python mirror reproduces it.
void captureDigest()
{
	World world = twoSidedWorld();
	const auto digestOf = [](const World & state) { return explorer_rl::snapshotDigest(explorer_rl::captureVisible(state)); };
	const auto base = digestOf(world);
	expect(base == digestOf(world), "the digest must be a function of the capture");
	world.find(3)->x = 5; // hidden enemy moves inside the fog
	world.blocking.insert({4, 4, 0}); // hidden tile changes
	expect(digestOf(world) == base, "hidden truth changed the digest");
	world.find(2)->owner = 0; // visible owner change
	expect(digestOf(world) != base, "a visible object change left the digest unchanged");
	world = twoSidedWorld();
	world.own.movement -= 100;
	expect(digestOf(world) != base, "an own-state change left the digest unchanged");
	world = twoSidedWorld();
	world.visible.erase({0, 0, 0});
	expect(digestOf(world) != base, "a visibility change left the digest unchanged");
	std::printf("capture digest %016" PRIx64 "\n", base);
}

void opponentKinds()
{
	using explorer_rl::opponentKind;
	expect(std::string(opponentKind(true, false, true)) == "hero", "a hero opponent must be reported as a hero");
	expect(std::string(opponentKind(false, true, false)) == "town_garrison", "a garrison must be reported as such");
	// Map monsters keep VCMI's unflaggable owner (-2), not the neutral color (-1).
	expect(std::string(opponentKind(false, false, true)) == "neutral", "an unowned army must be neutral");
	expect(std::string(opponentKind(false, false, false)) == "other", "an owned army without a hero is other");
}

int main()
{
	opponentKinds();
	baselineAndReconstruction();
	visibleToHiddenMove();
	transientSightingBetweenDecisions();
	hiddenAndVisibleRemoval();
	visibleOwnerChangeAndRehiding();
	independentPlayers();
	tileFactsFollowVisibility();
	ownStateChanges();
	randomizedReconstruction();
	captureDigest();
	std::cout << "visible-state tracker: 10 scenarios, 200 randomized folds, opponent kinds\n";
	return 0;
}
