// Real MetaString/Component parser tests. No game instance or fake callback.
#include "StdInc.h"
#include "Notice.h"

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

int main()
{
	resourcesAndMines();
	experienceAndWeeks();
	outcomes();
	malformedPoolsAndUnsupported();
	std::cout << "native notice parser: " << checks << " checks passed\n";
}
