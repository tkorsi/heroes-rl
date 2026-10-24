// Exercises the production notice parser on the operation sequences VCMI
// 1.7.5's audited producers build, and on near misses that must stay
// unsupported: other modes, components, identifiers, players and raw text.
#include "NoticeParser.h"

#include <cstdlib>
#include <iostream>

using explorer_rl::NoticeComponentInput;
using explorer_rl::NoticeDisplay;
using explorer_rl::NoticeOperation;
using explorer_rl::NoticeParse;
using explorer_rl::NoticeToken;
using explorer_rl::parseNotice;
using Kind = explorer_rl::NoticeOperationKind;
using Component = NoticeComponentInput;

namespace
{
int checks = 0;

[[noreturn]] void fail(const char * message)
{
	std::cerr << "FAILED: " << message << '\n';
	std::exit(1);
}

void expect(bool condition, const char * message)
{
	++checks;
	if(!condition)
		fail(message);
}

constexpr int RED = 0;
constexpr int BLUE = 1;

NoticeOperation op(Kind kind, const std::string & text = {}, int64_t number = 0)
{
	return {kind, text, number};
}

Component resource(Component::Kind kind, int resource, std::optional<int32_t> amount)
{
	return {kind, Component::Subtype::RESOURCE, resource, amount};
}

Component flag(int color)
{
	return {Component::Kind::FLAG, Component::Subtype::PLAYER, color, std::nullopt};
}

NoticeParse parse(NoticeDisplay mode, const std::vector<NoticeOperation> & operations,
	const std::vector<Component> & components = {}, int self = RED, int opponent = BLUE)
{
	return parseNotice(mode, operations, components, self, opponent);
}

bool unsupported(const NoticeParse & parse)
{
	return !parse.notice && !parse.unsupported.empty();
}

bool tokenIs(const NoticeToken & token, const std::string & op, std::optional<std::string> textID,
	std::optional<int64_t> number = std::nullopt, std::optional<std::string> owner = std::nullopt)
{
	return token.op == op && token.textID == textID && token.number == number && token.owner == owner;
}

void testResourcePickup()
{
	const std::vector<NoticeOperation> gold{op(Kind::APPEND_LOCAL_STRING, "core.advevent.113"), op(Kind::REPLACE_TEXTID_STRING, "core.restypes.6")};
	const auto parsed = parse(NoticeDisplay::INFO, gold, {resource(Component::Kind::RESOURCE, 6, 750)});
	expect(parsed.notice.has_value(), "CGResource's gold pickup must be recognized");
	const auto & notice = *parsed.notice;
	expect(notice.displayMode == "info", "The pickup keeps its native display mode");
	expect(notice.text.size() == 2 && tokenIs(notice.text[0], "append_text", "core.advevent.113")
		&& tokenIs(notice.text[1], "replace_text", "core.restypes.6"), "The pickup has exactly its two delivered operations");
	expect(notice.components.size() == 1 && notice.components[0].kind == "resource" && notice.components[0].resource == "gold"
		&& notice.components[0].amount == 750 && !notice.components[0].owner, "The pickup component carries resource and amount only");

	expect(unsupported(parse(NoticeDisplay::AUTO, gold, {resource(Component::Kind::RESOURCE, 6, 750)})), "A custom-message pickup mode is unaudited");
	expect(unsupported(parse(NoticeDisplay::INFO, gold, {resource(Component::Kind::RESOURCE, 5, 750)})), "Text and component must name one resource");
	expect(unsupported(parse(NoticeDisplay::INFO, gold, {resource(Component::Kind::RESOURCE, 6, std::nullopt)})), "A pickup needs its amount");
	expect(unsupported(parse(NoticeDisplay::INFO, gold, {resource(Component::Kind::RESOURCE_PER_DAY, 6, 750)})), "A pickup gives a resource, not income");
	expect(unsupported(parse(NoticeDisplay::INFO, gold, {resource(Component::Kind::RESOURCE, 6, 750), resource(Component::Kind::RESOURCE, 6, 1)})),
		"A surplus component is unsupported");
	expect(unsupported(parse(NoticeDisplay::INFO, gold)), "A missing component is unsupported");
	auto extra = gold;
	extra.push_back(op(Kind::REPLACE_NUMBER, {}, 750));
	expect(unsupported(parse(NoticeDisplay::INFO, extra, {resource(Component::Kind::RESOURCE, 6, 750)})), "The native gold notice has no number operation");
	expect(unsupported(parse(NoticeDisplay::INFO, {gold[0], op(Kind::REPLACE_TEXTID_STRING, "core.restypes.7")}, {resource(Component::Kind::RESOURCE, 7, 1)})),
		"Only the seven standard resources are audited");
	expect(unsupported(parse(NoticeDisplay::INFO, {gold[0], op(Kind::REPLACE_RAW_STRING, "core.restypes.6")}, {resource(Component::Kind::RESOURCE, 6, 1)})),
		"A raw string is never read as an identifier");
	expect(unsupported(parse(NoticeDisplay::AUTO, {op(Kind::APPEND_RAW_STRING, "You find a small pile of gold.")}, {resource(Component::Kind::RESOURCE, 6, 750)})),
		"A map-authored pickup message is unsupported");
}

void testMineFlag()
{
	const auto parsed = parse(NoticeDisplay::AUTO, {op(Kind::APPEND_TEXTID_STRING, "core.mineevnt.2")}, {resource(Component::Kind::RESOURCE_PER_DAY, 2, 2)});
	expect(parsed.notice.has_value(), "CGMine::flagMine must be recognized");
	expect(parsed.notice->displayMode == "auto" && parsed.notice->text.size() == 1 && tokenIs(parsed.notice->text[0], "append_text", "core.mineevnt.2"),
		"The mine notice has its one text operation");
	expect(parsed.notice->components.size() == 1 && parsed.notice->components[0].kind == "resource_per_day"
		&& parsed.notice->components[0].resource == "ore" && parsed.notice->components[0].amount == 2, "The mine notice carries its income");
	expect(unsupported(parse(NoticeDisplay::AUTO, {op(Kind::APPEND_TEXTID_STRING, "core.mineevnt.2")}, {resource(Component::Kind::RESOURCE_PER_DAY, 6, 1000)})),
		"Mine text and income must name one resource");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_TEXTID_STRING, "core.mineevnt.2")}, {resource(Component::Kind::RESOURCE_PER_DAY, 2, 2)})),
		"The mine notice mode is audited");
	expect(unsupported(parse(NoticeDisplay::AUTO, {op(Kind::APPEND_TEXTID_STRING, "core.mineevnt.02")}, {resource(Component::Kind::RESOURCE_PER_DAY, 2, 2)})),
		"Identifiers are matched exactly");
	expect(unsupported(parse(NoticeDisplay::AUTO, {op(Kind::APPEND_RAW_STRING, "A modded mine.")}, {resource(Component::Kind::RESOURCE_PER_DAY, 7, 1)})),
		"A described mod resource mine is unsupported");
}

void testExperienceLimit()
{
	const auto parsed = parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.genrltxt.1"), op(Kind::REPLACE_TEXTID_STRING, "hero.core.orrin.name")});
	expect(parsed.notice.has_value() && parsed.notice->text.size() == 2 && tokenIs(parsed.notice->text[1], "replace_text", "hero.core.orrin.name"),
		"The experience limit notice names the hero by identifier");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.genrltxt.1"), op(Kind::REPLACE_TEXTID_STRING, "hero.core.sorsha.name")})),
		"Only the scenario's hero names are audited");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.genrltxt.1"), op(Kind::REPLACE_RAW_STRING, "Orrin")})),
		"A raw hero name is unsupported");
}

void testVictoryAndLoss()
{
	// Red wins: Red is told its own message, Blue the message to others, both
	// with Red's name replacement and flag.
	const auto won = parse(NoticeDisplay::MODAL, {op(Kind::APPEND_RAW_STRING, "All opponents have been defeated."), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.0")},
		{flag(RED)}, RED, BLUE);
	expect(won.notice.has_value() && won.notice->text.size() == 1 && tokenIs(won.notice->text[0], "append_text", "heroes.baseline.all_opponents_defeated"),
		"Authored victory text maps exactly and its unused name replacement is omitted");
	expect(won.notice->components.size() == 1 && won.notice->components[0].kind == "flag" && won.notice->components[0].owner == "self"
		&& !won.notice->components[0].resource && !won.notice->components[0].amount, "The winner's flag is the recipient's own");
	const auto lost = parse(NoticeDisplay::MODAL, {op(Kind::APPEND_RAW_STRING, "Your opponent won."), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.0")},
		{flag(RED)}, BLUE, RED);
	expect(lost.notice.has_value() && tokenIs(lost.notice->text[0], "append_text", "heroes.baseline.opponent_won")
		&& lost.notice->components[0].owner == "opponent", "The loser sees the winner as its opponent");
	for(const auto & [authored, identifier] : explorer_rl::notice_detail::AUTHORED)
	{
		expect(std::string(authored).find('%') == std::string::npos, "Authored texts hold no placeholder");
		const auto parsed = parse(NoticeDisplay::MODAL, {op(Kind::APPEND_RAW_STRING, authored), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.1")}, {flag(BLUE)});
		expect(parsed.notice.has_value() && parsed.notice->text.size() == 1 && parsed.notice->text[0].textID == std::string(identifier),
			"Every authored victory or loss text maps to its identifier");
	}

	const auto standard = parse(NoticeDisplay::MODAL, {op(Kind::APPEND_TEXTID_STRING, "core.genrltxt.7"), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.1")},
		{flag(BLUE)}, BLUE, RED);
	expect(standard.notice.has_value() && standard.notice->text.size() == 2 && tokenIs(standard.notice->text[1], "replace_text", std::nullopt, std::nullopt, "self"),
		"A localized loss keeps its delivered name replacement as a relative owner");

	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_RAW_STRING, "All opponents have been defeated!"), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.0")}, {flag(RED)})),
		"Authored text must match exactly");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_TEXTID_STRING, "core.genrltxt.8"), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.0")}, {flag(RED)})),
		"Unaudited localized messages are unsupported");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_RAW_STRING, "Your opponent won."), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.2")}, {flag(2)})),
		"A third player is never relativized");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_RAW_STRING, "Your opponent won."), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.0")}, {flag(BLUE)})),
		"Name and flag must name one player");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_RAW_STRING, "Your opponent won.")}, {flag(RED)})),
		"The name replacement is part of the audited shape");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_RAW_STRING, "Your opponent won."), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.0")})),
		"The flag is part of the audited shape");
	expect(unsupported(parse(NoticeDisplay::INFO, {op(Kind::APPEND_RAW_STRING, "Your opponent won."), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.0")}, {flag(RED)})),
		"The victory notice mode is audited");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_TEXTID_STRING, "vcmi.server.errors.playerLeft"), op(Kind::REPLACE_TEXTID_STRING, "vcmi.capitalColors.1")}, {flag(BLUE)})),
		"A disconnect notice is unsupported");
}

void testNewWeek()
{
	const auto week = parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.133"), op(Kind::REPLACE_LOCAL_STRING, "core.arraytxt.43")});
	expect(week.notice.has_value() && week.notice->text.size() == 2 && tokenIs(week.notice->text[1], "replace_text", "core.arraytxt.43") && week.notice->components.empty(),
		"An ordinary week is recognized with its random name");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.133"), op(Kind::REPLACE_LOCAL_STRING, "core.arraytxt.42")})),
		"Week names come only from the engine's range");
	expect(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.130"), op(Kind::REPLACE_LOCAL_STRING, "core.arraytxt.41")}).notice.has_value(),
		"A new month is recognized with its random name");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.130"), op(Kind::REPLACE_LOCAL_STRING, "core.arraytxt.43")})),
		"Month names come only from the engine's range");
	const std::string imp = "creatures.core.imp.name.singular";
	const auto doubled = parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.131"), op(Kind::REPLACE_TEXTID_STRING, imp), op(Kind::REPLACE_TEXTID_STRING, imp)});
	expect(doubled.notice.has_value() && doubled.notice->text.size() == 3, "A creature week names its creature twice");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.134"), op(Kind::REPLACE_TEXTID_STRING, imp),
		op(Kind::REPLACE_TEXTID_STRING, "creatures.core.familiar.name.singular")})), "Both names are the week's one creature");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.131"), op(Kind::REPLACE_TEXTID_STRING, "creatures.core.imp.name.plural"),
		op(Kind::REPLACE_TEXTID_STRING, "creatures.core.imp.name.plural")})), "The week uses singular names");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.131"), op(Kind::REPLACE_TEXTID_STRING, "creature.core.imp.name.singular"),
		op(Kind::REPLACE_TEXTID_STRING, "creature.core.imp.name.singular")})), "The creature namespace is plural");
	expect(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.132")}).notice.has_value(), "A plague week is recognized");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.132"), op(Kind::APPEND_EOL)})), "A surplus operation is unsupported");
	for(const int64_t growth : {int64_t{5}, int64_t{0}, int64_t{-3}})
	{
		const auto deity = parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.135"), op(Kind::REPLACE_TEXTID_STRING, imp),
			op(Kind::REPLACE_TEXTID_STRING, imp), op(Kind::REPLACE_POSITIVE_NUMBER, {}, growth), op(Kind::REPLACE_TEXTID_STRING, "creatures.core.familiar.name.singular"),
			op(Kind::REPLACE_POSITIVE_NUMBER, {}, growth)});
		expect(deity.notice.has_value() && deity.notice->text.size() == 6 && tokenIs(deity.notice->text[3], "replace_positive_number", std::nullopt, growth),
			"Deity of fire keeps signed growth numbers");
	}
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.135"), op(Kind::REPLACE_TEXTID_STRING, imp),
		op(Kind::REPLACE_TEXTID_STRING, imp), op(Kind::REPLACE_POSITIVE_NUMBER, {}, 5), op(Kind::REPLACE_TEXTID_STRING, imp),
		op(Kind::REPLACE_POSITIVE_NUMBER, {}, 6)})), "Deity of fire repeats one growth");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.133"), op(Kind::REPLACE_LOCAL_STRING, "core.arraytxt.43")},
		{resource(Component::Kind::RESOURCE, 6, 1)})), "A new week notice has no components");
	expect(unsupported(parse(NoticeDisplay::INFO, {op(Kind::APPEND_LOCAL_STRING, "core.arraytxt.133"), op(Kind::REPLACE_LOCAL_STRING, "core.arraytxt.43")})),
		"The new week notice mode is audited");
}

void testUnauditedTemplates()
{
	expect(unsupported(parse(NoticeDisplay::MODAL, {})), "An empty notice is unsupported");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_EOL)})), "A bare line end is unsupported");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_NUMBER, {}, 3)})), "A bare number is unsupported");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_LOCAL_STRING, "core.advevent.112")})), "Nearby identifiers are unsupported");
	expect(unsupported(parse(NoticeDisplay::MODAL, {op(Kind::APPEND_TEXTID_STRING, "core.advevent.113"), op(Kind::REPLACE_TEXTID_STRING, "core.restypes.6")},
		{resource(Component::Kind::RESOURCE, 6, 1)})), "The pickup text is a local string, not a text identifier");
}
}

int main()
{
	testResourcePickup();
	testMineFlag();
	testExperienceLimit();
	testVictoryAndLoss();
	testNewWeek();
	testUnauditedTemplates();
	std::cout << "notice parser: 6 groups, " << checks << " checks passed\n";
	return 0;
}
