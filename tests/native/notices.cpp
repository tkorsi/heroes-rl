// Real MetaString/Component parser tests. No game instance or fake callback.
#include "StdInc.h"
#include "Notice.h"
#include "lib/GameLibrary.h"
#include "lib/CCreatureHandler.h"
#include "lib/entities/artifact/CArtHandler.h"
#include "lib/spells/CSpellHandler.h"
#include "lib/texts/CGeneralTextHandler.h"

#include <array>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>

namespace
{
int checks = 0;

void expect(bool condition, const char * message)
{
	++checks;
	if(!condition)
	{
		std::cerr << "FAILED: " << message << '\n';
		std::exit(1);
	}
}

void reject(const std::function<void()> & function, const char * message)
{
	try { function(); }
	catch(const explorer_rl::UnsupportedNotice &) { ++checks; return; }
	expect(false, message);
}

JsonNode parse(EInfoWindowMode mode, const MetaString & text, const std::vector<Component> & components = {}, int recipient = 0)
{
	return explorer_rl::normalizeNotice(mode, text, components, PlayerColor(recipient), {"hero.core.orrin.name"});
}

MetaString resourceText(int resource = 6)
{
	MetaString text;
	text.appendLocalString(EMetaText::ADVOB_TXT, 113);
	text.replaceTextID("core.restypes." + std::to_string(resource));
	return text;
}

void resourcesAndMines()
{
	const std::vector<Component> gold{Component(ComponentType::RESOURCE, GameResID(6), 500)};
	const auto result = parse(EInfoWindowMode::INFO, resourceText(), gold);
	expect(result["display_mode"].String() == "info", "Resource mode");
	expect(result["text"].Vector().size() == 2, "Gold contains no invented quantity replacement");
	expect(result["components"][0]["resource"].String() == "gold", "Resource identity");
	expect(result["components"][0]["amount"].Integer() == 500, "Resource quantity");
	for(int resource = 0; resource < 7; ++resource)
	{
		const auto text = MetaString::createFromTextID("core.mineevnt." + std::to_string(resource));
		const auto mine = parse(EInfoWindowMode::AUTO, text,
			{Component(ComponentType::RESOURCE_PER_DAY, GameResID(resource), 2)});
		expect(mine["components"][0]["kind"].String() == "resource_per_day", "Mine resource component");
	}
	reject([&] { parse(EInfoWindowMode::INFO, resourceText(0), gold); }, "Contradictory resource component");
	reject([&] { parse(EInfoWindowMode::MODAL, resourceText(), gold); }, "Unverified resource mode");
	reject([&] { parse(EInfoWindowMode::INFO, resourceText(), {}); }, "Missing resource component");
	reject([&] { parse(EInfoWindowMode::INFO, resourceText(),
		{Component(ComponentType::RESOURCE, PlayerColor(6), 500)}); }, "Wrong identifier variant");
	auto extra = resourceText();
	extra.replaceNumber(500);
	reject([&] { parse(EInfoWindowMode::INFO, extra, gold); }, "Unused number operation");
}

void experienceAndWeeks()
{
	MetaString cap;
	cap.appendLocalString(EMetaText::GENERAL_TXT, 1);
	cap.replaceTextID("hero.core.orrin.name");
	expect(parse(EInfoWindowMode::MODAL, cap)["text"][1]["text_id"].String() == "hero.core.orrin.name", "Own hero cap notice");
	reject([&] { explorer_rl::normalizeNotice(EInfoWindowMode::MODAL, cap, {}, PlayerColor(0), {}); }, "Unowned hero cap notice");
	for(const auto [base, first, last] : {std::array<int, 3>{130, 32, 41}, std::array<int, 3>{133, 43, 57}})
		for(int name = first; name <= last; ++name)
		{
			MetaString week;
			week.appendLocalString(EMetaText::ARRAY_TXT, base);
			week.replaceLocalString(EMetaText::ARRAY_TXT, name);
			expect(parse(EInfoWindowMode::MODAL, week)["text"].Vector().size() == 2, "Ordinary new-week or month");
		}
	MetaString plague;
	plague.appendLocalString(EMetaText::ARRAY_TXT, 132);
	expect(parse(EInfoWindowMode::MODAL, plague)["components"].Vector().empty(), "Plague has no components");
	for(int kind : {131, 134, 135})
	{
		MetaString week;
		week.appendLocalString(EMetaText::ARRAY_TXT, kind);
		week.replaceTextID("creatures.core.imp.name.singular");
		week.replaceTextID("creatures.core.imp.name.singular");
		if(kind == 135)
		{
			week.replacePositiveNumber(std::numeric_limits<int64_t>::min());
			week.replaceTextID("creatures.core.familiar.name.singular");
			week.replacePositiveNumber(std::numeric_limits<int64_t>::min());
		}
		const auto result = parse(EInfoWindowMode::MODAL, week);
		expect(result["text"].Vector().size() == (kind == 135 ? 6 : 3), "Creature week grammar");
		if(kind == 135)
			expect(result["text"][3]["number"].Integer() == std::numeric_limits<int64_t>::min(), "Signed growth is not narrowed");
	}
	MetaString mismatched;
	mismatched.appendLocalString(EMetaText::ARRAY_TXT, 131);
	mismatched.replaceTextID("creatures.core.imp.name.singular");
	mismatched.replaceTextID("creatures.core.familiar.name.singular");
	reject([&] { parse(EInfoWindowMode::MODAL, mismatched); }, "Mismatched same-creature replacements");
	MetaString plural;
	plural.appendLocalString(EMetaText::ARRAY_TXT, 131);
	plural.replaceTextID("creatures.core.imp.name.plural");
	plural.replaceTextID("creatures.core.imp.name.plural");
	reject([&] { parse(EInfoWindowMode::MODAL, plural); }, "Unaudited plural operation sequence");
}

void outcomes()
{
	const std::array<std::pair<const char *, const char *>, 4> messages{{
		{"All opponents have been defeated.", "heroes.baseline.all_opponents_defeated"},
		{"Your opponent won.", "heroes.baseline.opponent_won"},
		{"Seven days without a town have passed.", "heroes.baseline.seven_days_without_town"},
		{"Your opponent lost their last town.", "heroes.baseline.opponent_lost_last_town"}}};
	for(const auto & [raw, id] : messages)
	{
		auto text = MetaString::createFromRawString(raw);
		text.replaceTextID("vcmi.capitalColors.1");
		for(int recipient : {0, 1})
		{
			const auto result = parse(EInfoWindowMode::MODAL, text, {Component(ComponentType::FLAG, PlayerColor(1))}, recipient);
			expect(result["text"].Vector().size() == 1, "Drop only audited unused result replacement");
			expect(result["text"][0]["text_id"].String() == id, "Exact authored result normalized");
			expect(result["components"][0]["owner"].String() == (recipient == 1 ? "self" : "opponent"), "Relative flag");
		}
	}
	for(int id : {5, 7, 8, 659})
	{
		auto text = MetaString::createFromTextID("core.genrltxt." + std::to_string(id));
		text.replaceTextID("vcmi.capitalColors.0");
		const auto result = parse(EInfoWindowMode::MODAL, text, {Component(ComponentType::FLAG, PlayerColor(0))});
		expect(result["text"].Vector().size() == (id == 659 ? 1 : 2), "Only victory template has unused replacement");
		if(id != 659)
			expect(result["text"][1]["owner"].String() == "self", "Displayed player replacement preserved relatively");
	}
	auto text = MetaString::createFromRawString(messages[0].first);
	text.replaceTextID("vcmi.capitalColors.0");
	reject([&] { parse(EInfoWindowMode::MODAL, text, {Component(ComponentType::FLAG, PlayerColor(1))}); }, "Contradictory flag");
	auto hidden = MetaString::createFromRawString(messages[0].first);
	hidden.replaceTextID("hero.core.edric.name");
	reject([&] { parse(EInfoWindowMode::MODAL, hidden, {Component(ComponentType::FLAG, PlayerColor(0))}); }, "Unused arbitrary replacement");
	auto near = MetaString::createFromRawString("All opponents have been defeated. ");
	near.replaceTextID("vcmi.capitalColors.0");
	reject([&] { parse(EInfoWindowMode::MODAL, near, {Component(ComponentType::FLAG, PlayerColor(0))}); }, "Near-match raw result");
}

void trophies()
{
	// Static content only: no game state, server, client, map or callbacks.
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	LIBRARY->initializeLibrary();
	MetaString text;
	text.appendLocalString(EMetaText::GENERAL_TXT, 30);
	const ArtifactID artifact(ArtifactID::decode("core:centaurAxe"));
	const SpellID spell(SpellID::decode("core:magicArrow"));
	const std::vector<Component> components{
		Component(ComponentType::ARTIFACT, artifact),
		Component(ComponentType::SPELL_SCROLL, spell),
		Component(ComponentType::ARTIFACT, artifact)};
	for(int recipient : {0, 1})
	{
		const auto notice = parse(EInfoWindowMode::MODAL, text, components, recipient);
		expect(notice["display_mode"].String() == "modal", "Native trophy presentation mode");
		expect(notice["text"].Vector().size() == 1 && notice["text"][0]["text_id"].String() == "core.genrltxt.30",
			"Exact native trophy template");
		expect(notice["components"].Vector().size() == 3, "Preserve trophy order and duplicates");
		expect(notice["components"][0]["kind"].String() == "artifact"
			&& notice["components"][0]["content_id"].String() == "core:centaurAxe", "Static artifact identity");
		expect(notice["components"][1]["kind"].String() == "spell_scroll"
			&& notice["components"][1]["content_id"].String() == "core:magicArrow", "Displayed scroll spell identity");
		expect(notice["components"][2] == notice["components"][0], "Repeated trophy preserved");
		for(const auto & component : notice["components"].Vector())
			expect(component.Struct().size() == 2, "No instance ID, inventory, quantity or owner added");
	}
	for(auto mode : {EInfoWindowMode::INFO, EInfoWindowMode::AUTO})
		reject([&] { parse(mode, text, components); }, "Wrong trophy presentation mode");
	reject([&] { parse(EInfoWindowMode::MODAL, text); }, "Empty trophy list");
	reject([&] { parse(EInfoWindowMode::MODAL, MetaString::createFromTextID("core.genrltxt.30"), components); },
		"Text-ID operation is not native local-string operation");
	auto extra = text;
	extra.replaceNumber(7);
	reject([&] { parse(EInfoWindowMode::MODAL, extra, components); }, "Unused trophy operation");
	for(const auto & bad : std::vector<Component>{
		Component(ComponentType::ARTIFACT, artifact, 1),
		Component(ComponentType::SPELL_SCROLL, spell, 0),
		Component(ComponentType::ARTIFACT, spell),
		Component(ComponentType::SPELL_SCROLL, artifact),
		Component(ComponentType::ARTIFACT, ArtifactID(-1)),
		Component(ComponentType::ARTIFACT, ArtifactID(ArtifactID::SPELL_SCROLL)),
		Component(ComponentType::ARTIFACT, ArtifactID(std::numeric_limits<int32_t>::max())),
		Component(ComponentType::SPELL_SCROLL, SpellID(SpellID::PRESET)),
		Component(ComponentType::SPELL_SCROLL, SpellID(std::numeric_limits<int32_t>::max())),
		Component(ComponentType::SPELL, spell),
		Component(ComponentType::RESOURCE, GameResID(6), 500)})
	{
		auto mixed = components;
		mixed.push_back(bad);
		reject([&] { parse(EInfoWindowMode::MODAL, text, mixed); }, "Reject whole trophy list with invalid component");
	}
	reject([&] { parse(EInfoWindowMode::MODAL, text, std::vector<Component>(33, components[0])); }, "Bound trophy list");
	// The real static spell object permits metadata mutation. Malformed fixture
	// keys exercise rejection only; no game is created and originals are restored.
	auto nativeSpell = LIBRARY->spellh->objects.at(spell.getNum());
	const auto scope = nativeSpell->modScope;
	const auto name = nativeSpell->identifier;
	for(const auto & [badScope, badName] : std::vector<std::pair<std::string, std::string>>{
		{"mod", name}, {"core", "1numeric"}, {"core", "../../private"}, {"core", ""},
		{"core", std::string(65, 'a')}, {"core", "magicArrow:17"}})
	{
		nativeSpell->modScope = badScope;
		nativeSpell->identifier = badName;
		reject([&] { parse(EInfoWindowMode::MODAL, text, components); }, "Reject non-core or malformed static content key");
	}
	nativeSpell->modScope = scope;
	nativeSpell->identifier = name;
	LIBRARY->spellh->objects.at(spell.getNum()).reset();
	reject([&] { parse(EInfoWindowMode::MODAL, text, components); }, "Reject missing static content entry");
	LIBRARY->spellh->objects.at(spell.getNum()) = nativeSpell;
}

void necromancy()
{
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	LIBRARY->initializeLibrary();
	const CreatureID skeleton(CreatureID::decode("core:skeleton"));
	const auto * raised = LIBRARY->creatures()->getById(skeleton);
	const auto textFor = [&](int count)
	{
		MetaString text;
		text.appendLocalString(EMetaText::GENERAL_TXT, count > 1 ? 145 : 146);
		if(count > 1) text.replaceNumber(count);
		text.replaceTextID(count > 1 ? raised->getNamePluralTextID() : raised->getNameSingularTextID());
		return text;
	};
	const auto parseNecromancy = [&](EInfoWindowMode mode, const MetaString & text,
		const std::vector<Component> & components, bool dialogs = true)
	{
		return explorer_rl::normalizeNotice(mode, text, components, PlayerColor(0), {}, dialogs);
	};
	for(int count : {1, 4, std::numeric_limits<int32_t>::max()})
	{
		const auto text = textFor(count);
		const std::vector<Component> components{Component(ComponentType::CREATURE, skeleton, count)};
		for(int player : {0, 1})
		{
			const auto notice = explorer_rl::normalizeNotice(EInfoWindowMode::AUTO, text, components, PlayerColor(player), {}, true);
			expect(notice["display_mode"].String() == "auto", "Necromancy is an automatic notice");
			expect(notice["text"].Vector().size() == (count > 1 ? 3 : 2), "Exact native singular/plural tokens");
			expect(notice["components"][0]["kind"].String() == "creature"
				&& notice["components"][0]["content_id"].String() == "core:skeleton", "Displayed static creature type");
			expect(notice["components"][0]["amount"].Integer() == count, "Displayed raised count");
			if(count > 1)
				expect(notice["text"][1]["op"].String() == "replace_number"
					&& notice["text"][1]["number"].Integer() == count, "Exact number substitution");
		}
		reject([&] { parseNecromancy(EInfoWindowMode::AUTO, text, components, false); }, "Legacy profile unchanged");
		for(auto mode : {EInfoWindowMode::MODAL, EInfoWindowMode::INFO})
			reject([&] { parseNecromancy(mode, text, components); }, "Wrong necromancy mode");
	}
	const auto text = textFor(4);
	const std::vector<Component> components{Component(ComponentType::CREATURE, skeleton, 4)};
	for(const auto & bad : std::vector<Component>{
		Component(ComponentType::CREATURE, skeleton),
		Component(ComponentType::CREATURE, skeleton, 0),
		Component(ComponentType::CREATURE, skeleton, -1),
		Component(ComponentType::CREATURE, skeleton, 3),
		Component(ComponentType::CREATURE, CreatureID(0), 4),
		Component(ComponentType::CREATURE, CreatureID(-1), 4),
		Component(ComponentType::CREATURE, CreatureID(std::numeric_limits<int32_t>::max()), 4),
		Component(ComponentType::CREATURE, GameResID(skeleton.getNum()), 4),
		Component(ComponentType::RESOURCE, GameResID(6), 4)})
		reject([&] { parseNecromancy(EInfoWindowMode::AUTO, text, {bad}); }, "Malformed or contradictory raised component");
	reject([&] { parseNecromancy(EInfoWindowMode::AUTO, text, {}); }, "Missing raised component");
	reject([&] { parseNecromancy(EInfoWindowMode::AUTO, text, {components[0], components[0]}); }, "Extra raised component");
	reject([&] { parseNecromancy(EInfoWindowMode::AUTO, textFor(1), components); }, "Singular/count mismatch");
	auto extra = text;
	extra.replaceTextID("hero.core.thant.name");
	reject([&] { parseNecromancy(EInfoWindowMode::AUTO, extra, components); }, "Undisplayed hero substitution");
	MetaString wrongName;
	wrongName.appendLocalString(EMetaText::GENERAL_TXT, 145);
	wrongName.replaceNumber(4);
	wrongName.replaceTextID(raised->getNameSingularTextID());
	reject([&] { parseNecromancy(EInfoWindowMode::AUTO, wrongName, components); }, "Wrong number agreement in name");
	MetaString wrongNumber;
	wrongNumber.appendLocalString(EMetaText::GENERAL_TXT, 145);
	wrongNumber.replacePositiveNumber(4);
	wrongNumber.replaceTextID(raised->getNamePluralTextID());
	reject([&] { parseNecromancy(EInfoWindowMode::AUTO, wrongNumber, components); }, "Wrong numeric operation");
}

void capabilities()
{
	// These are actual native MetaString/Component values, normalized with the
	// static VCMI catalog only. No world, native callback or game is fabricated.
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	LIBRARY->initializeLibrary();
	const std::set<std::string> own{"hero.core.orrin.name"};
	const auto expanded = [&](EInfoWindowMode mode, const MetaString & text,
		const std::vector<Component> & components = std::vector<Component>{}, int player = 0)
	{
		return explorer_rl::normalizeNotice(mode, text, components, PlayerColor(player), own, true, true);
	};
	const auto unusedPools = [&](const MetaString & text, const std::vector<Component> & components)
	{
		JsonNode original;
		text.jsonSerialize(original);
		for(const char * pool : {"exactStrings", "localStrings", "stringsTextID", "numbers"})
		{
			auto data = original;
			data[pool].Vector().push_back(pool == std::string("localStrings") || pool == std::string("numbers")
				? JsonNode(int64_t(17)) : JsonNode("hidden data"));
			MetaString forged;
			forged.jsonDeserialize(data);
			reject([&] { expanded(EInfoWindowMode::MODAL, forged, components); }, "Capability unused pool rejected");
		}
	};
	for(int id : {123, 124, 125, 135, 335, 580})
	{
		MetaString text;
		text.appendLocalString(EMetaText::GENERAL_TXT, id);
		for(int player : {0, 1})
		{
			const auto result = expanded(EInfoWindowMode::MODAL, text, {}, player);
			expect(result["display_mode"].String() == "modal", "Capability exact display mode");
			expect(result["text"].Vector().size() == 1
				&& result["text"][0]["text_id"].String() == "core.genrltxt." + std::to_string(id), "Capability exact template");
			expect(result["components"].Vector().empty(), "Informational template creates no reward component");
		}
		for(auto mode : {EInfoWindowMode::AUTO, EInfoWindowMode::INFO})
			reject([&] { expanded(mode, text); }, "Capability wrong display mode");
		reject([&] { explorer_rl::normalizeNotice(EInfoWindowMode::MODAL, text, {}, PlayerColor(0), own, true, false); },
			"Capability template cannot expand a legacy profile");
		reject([&] { expanded(EInfoWindowMode::MODAL, text, {Component(ComponentType::RESOURCE, GameResID(6), 400)}); },
			"Capability template cannot invent resource reward");
		reject([&] { expanded(EInfoWindowMode::MODAL, MetaString::createFromTextID("core.genrltxt." + std::to_string(id))); },
			"Capability native operation must match");
		unusedPools(text, {});
	}
	MetaString summon;
	summon.appendLocalString(EMetaText::GENERAL_TXT, 336);
	summon.replaceRawString(LIBRARY->generaltexth->translate(*own.begin()));
	const auto summonResult = expanded(EInfoWindowMode::MODAL, summon);
	expect(summonResult["text"].Vector().size() == 2
		&& summonResult["text"][1]["text_id"].String() == *own.begin(), "Owned caster localized name becomes its allowed text ID");
	reject([&] { explorer_rl::normalizeNotice(EInfoWindowMode::MODAL, summon, {}, PlayerColor(0), {}, true, true); },
		"Summon caster must be owned");
	MetaString foreign;
	foreign.appendLocalString(EMetaText::GENERAL_TXT, 336);
	foreign.replaceRawString("Unrelated hidden hero");
	reject([&] { expanded(EInfoWindowMode::MODAL, foreign); }, "Arbitrary raw caster name rejected");
	unusedPools(summon, {});
	MetaString dimensionDoor;
	dimensionDoor.appendTextID("core.genrltxt.70");
	expect(expanded(EInfoWindowMode::MODAL, dimensionDoor)["text"][0]["text_id"].String() == "core.genrltxt.70",
		"Dimension Door native append-text operation preserved");
	MetaString badDimensionDoor;
	badDimensionDoor.appendLocalString(EMetaText::GENERAL_TXT, 70);
	reject([&] { expanded(EInfoWindowMode::MODAL, badDimensionDoor); }, "Dimension Door append-local is not native grammar");
	MetaString scuttle;
	scuttle.appendTextID("core.genrltxt.337");
	scuttle.replaceRawString(LIBRARY->generaltexth->translate(*own.begin()));
	expect(expanded(EInfoWindowMode::MODAL, scuttle)["text"][1]["text_id"].String() == *own.begin(),
		"Scuttle owned caster localized name normalized");
	MetaString badScuttle;
	badScuttle.appendLocalString(EMetaText::GENERAL_TXT, 337);
	badScuttle.replaceRawString(LIBRARY->generaltexth->translate(*own.begin()));
	reject([&] { expanded(EInfoWindowMode::MODAL, badScuttle); }, "Scuttle append-local is not native grammar");
	for(const auto & text : {dimensionDoor, scuttle})
	{
		for(auto mode : {EInfoWindowMode::AUTO, EInfoWindowMode::INFO})
			reject([&] { expanded(mode, text); }, "Casting failure wrong display mode");
		reject([&] { explorer_rl::normalizeNotice(EInfoWindowMode::MODAL, text, {}, PlayerColor(0), own, true, false); },
			"Casting failure cannot expand legacy profiles");
		unusedPools(text, {});
	}
	reject([&] { explorer_rl::normalizeNotice(EInfoWindowMode::MODAL, scuttle, {}, PlayerColor(0), {}, true, true); },
		"Scuttle caster must be owned");
	std::vector<SpellID> spells;
	for(const char * name : {"core:magicArrow", "core:haste", "core:slow"}) spells.emplace_back(SpellID::decode(name));
	std::sort(spells.begin(), spells.end());
	const auto eagle = [&](const std::vector<SpellID> & ids, bool period = true, const std::string & hero = "hero.core.orrin.name")
	{
		MetaString text;
		text.appendLocalString(EMetaText::GENERAL_TXT, 221);
		text.replaceTextID(hero);
		for(size_t i = 0; i < ids.size(); ++i)
		{
			const auto name = LIBRARY->spells()->getById(ids[i])->getNameTextID();
			if(i == 0) text.replaceTextID(name);
			else
			{
				if(i + 1 == ids.size()) text.appendLocalString(EMetaText::GENERAL_TXT, 141);
				else text.appendRawString(", ");
				text.appendTextID(name);
			}
		}
		if(period) text.appendRawString(".");
		return text;
	};
	for(size_t count : {1, 2, 3})
	{
		const std::vector<SpellID> selected(spells.begin(), spells.begin() + count);
		const auto text = eagle(selected);
		std::vector<Component> components;
		for(const auto spell : selected) components.emplace_back(ComponentType::SPELL, spell);
		for(int player : {0, 1})
		{
			const auto result = expanded(EInfoWindowMode::MODAL, text, components, player);
			expect(result["text"].Vector().size() == 2 * count + 2, "Native Eagle Eye text order preserved");
			expect(result["components"].Vector().size() == count, "Native Eagle Eye spell count preserved");
			for(size_t i = 0; i < count; ++i)
			{
				expect(result["components"][i]["kind"].String() == "spell"
					&& result["components"][i]["content_id"].String() == LIBRARY->spells()->getById(selected[i])->getJsonKey(),
					"Native learned spell type and order preserved");
				expect(result["components"][i]["amount"].isNull() && result["components"][i]["owner"].isNull(),
					"Eagle Eye adds no inferred quantity or owner");
			}
			expect(result["text"][2 * count + 1]["text_id"].String() == "heroes.notice.period", "Native period normalized");
			if(count == 3) expect(result["text"][3]["text_id"].String() == "heroes.notice.comma_space", "Native comma normalized");
			if(count > 1) expect(result["text"][2 * count - 1]["text_id"].String() == "core.genrltxt.141", "Native final conjunction preserved");
		}
		for(auto mode : {EInfoWindowMode::INFO, EInfoWindowMode::AUTO})
			reject([&] { expanded(mode, text, components); }, "Eagle Eye wrong mode rejected");
		reject([&] { expanded(EInfoWindowMode::MODAL, eagle(selected, false), components); }, "Eagle Eye missing final punctuation");
		reject([&] { expanded(EInfoWindowMode::MODAL, eagle(selected, true, "hero.core.edric.name"), components); }, "Eagle Eye foreign hero rejected");
		reject([&] { expanded(EInfoWindowMode::MODAL, text, {}); }, "Eagle Eye missing learned spells");
		reject([&] { explorer_rl::normalizeNotice(EInfoWindowMode::MODAL, text, components, PlayerColor(0), own, true, false); },
			"Eagle Eye cannot expand legacy profiles");
		auto extra = components;
		extra.push_back(Component(ComponentType::SPELL, selected.front(), 0));
		reject([&] { expanded(EInfoWindowMode::MODAL, text, extra); }, "Eagle Eye unrendered spell component rejected");
		if(count > 1)
		{
			auto reversed = components;
			std::reverse(reversed.begin(), reversed.end());
			reject([&] { expanded(EInfoWindowMode::MODAL, text, reversed); }, "Eagle Eye component order mismatch rejected");
		}
		unusedPools(text, components);
	}
	const auto single = eagle({spells[0]});
	for(const auto & malformed : std::vector<Component>{Component(ComponentType::SPELL, spells[0], 0),
		Component(ComponentType::SPELL, ArtifactID(spells[0].getNum())), Component(ComponentType::SPELL, SpellID(-1)),
		Component(ComponentType::SPELL_SCROLL, spells[0]), Component(ComponentType::SPELL, spells[1])})
		reject([&] { expanded(EInfoWindowMode::MODAL, single, {malformed}); }, "Eagle Eye malformed or mismatched spell rejected");
	reject([&] { expanded(EInfoWindowMode::MODAL, eagle({spells[0], spells[0]}),
		{Component(ComponentType::SPELL, spells[0]), Component(ComponentType::SPELL, spells[0])}); }, "Eagle Eye duplicate spell rejected");
}

void malformedPoolsAndUnsupported()
{
	const std::vector<Component> gold{Component(ComponentType::RESOURCE, GameResID(6), 500)};
	MetaString alias;
	alias.appendLocalString(EMetaText::GENERAL_TXT, 20113);
	alias.replaceTextID("core.restypes.6");
	reject([&] { parse(EInfoWindowMode::INFO, alias, gold); }, "Large local index must not alias an allowed text family");
	JsonNode original;
	resourceText().jsonSerialize(original);
	for(const char * name : {"exactStrings", "localStrings", "stringsTextID", "numbers"})
	{
		auto data = original;
		data[name].Vector().push_back(name == std::string("localStrings") || name == std::string("numbers")
			? JsonNode(int64_t(42)) : JsonNode("hidden data"));
		MetaString forged;
		forged.jsonDeserialize(data);
		reject([&] { parse(EInfoWindowMode::INFO, forged, gold); }, "Unused pool must not survive");
	}
	{
		auto data = original;
		data["stringsTextID"].Vector().clear();
		MetaString forged;
		forged.jsonDeserialize(data);
		reject([&] { parse(EInfoWindowMode::INFO, forged, gold); }, "Missing used pool value");
	}
	{
		auto data = original;
		data["message"][0].Integer() = 99;
		MetaString forged;
		forged.jsonDeserialize(data);
		reject([&] { parse(EInfoWindowMode::INFO, forged, gold); }, "Unknown operation code");
	}
	for(int id : {30, 145, 146, 221})
	{
		MetaString postBattle;
		postBattle.appendLocalString(EMetaText::GENERAL_TXT, id);
		reject([&] { parse(EInfoWindowMode::MODAL, postBattle); }, "Unaudited post-battle template");
	}
	reject([&] { parse(EInfoWindowMode::INFO, resourceText(), gold, 2); }, "Unsupported recipient");
	reject([&] { parse(static_cast<EInfoWindowMode>(99), resourceText(), gold); }, "Unknown display mode");
	reject([&] { parse(EInfoWindowMode::INFO, MetaString(), gold); }, "Empty notice");
}
}

int main(int argc, char ** argv)
{
	if(argc == 2 && std::string(argv[1]) == "--capabilities")
	{
		capabilities();
		std::cout << "native capability notice parser: " << checks << " checks passed\n";
		return 0;
	}
	if(argc == 2 && std::string(argv[1]) == "--necromancy")
	{
		necromancy();
		std::cout << "native necromancy notice parser: " << checks << " checks passed\n";
		return 0;
	}
	if(argc == 2 && std::string(argv[1]) == "--trophies")
	{
		trophies();
		std::cout << "native trophy notice parser: " << checks << " checks passed\n";
		return 0;
	}
	resourcesAndMines();
	experienceAndWeeks();
	outcomes();
	malformedPoolsAndUnsupported();
	std::cout << "native notice parser: " << checks << " checks passed\n";
}
