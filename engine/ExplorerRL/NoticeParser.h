#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/// Maps one informational notification, as the server built it, onto the
/// contextual history's typed notice. Only the message templates audited in
/// VCMI 1.7.5's producers for the baseline scenario are recognized, each by its
/// complete operation sequence, every pool entry, display mode and component
/// list; anything else is unsupported, and nothing is reconstructed from
/// rendered text. This header has no engine dependency: ExplorerRL copies
/// MetaString::operations() and the components into these inputs, and tests
/// build them directly.
namespace explorer_rl
{
/// MetaStringOperation::Kind, in the same order.
enum class NoticeOperationKind : uint8_t
{
	APPEND_RAW_STRING,
	APPEND_LOCAL_STRING,
	APPEND_TEXTID_STRING,
	APPEND_NUMBER,
	REPLACE_RAW_STRING,
	REPLACE_LOCAL_STRING,
	REPLACE_TEXTID_STRING,
	REPLACE_NUMBER,
	REPLACE_POSITIVE_NUMBER,
	APPEND_EOL
};

struct NoticeOperation
{
	NoticeOperationKind kind = NoticeOperationKind::APPEND_EOL;
	std::string text; ///< raw string or text identifier; local strings as their identifier
	int64_t number = 0;
};

/// EInfoWindowMode, in the same order.
enum class NoticeDisplay : uint8_t
{
	AUTO,
	MODAL,
	INFO
};

/// The parts of a Component the audited templates use. The subtype kind is the
/// identifier type the component's variant holds.
struct NoticeComponentInput
{
	enum class Kind : uint8_t
	{
		RESOURCE,
		RESOURCE_PER_DAY,
		FLAG,
		OTHER
	};
	enum class Subtype : uint8_t
	{
		RESOURCE,
		PLAYER,
		OTHER
	};

	Kind kind = Kind::OTHER;
	Subtype subtype = Subtype::OTHER;
	int32_t number = -1; ///< the subtype's numeric identifier
	std::optional<int32_t> value;
};

struct NoticeToken
{
	std::string op; ///< append_text, replace_text, append_number, replace_number, replace_positive_number or newline
	std::optional<std::string> textID;
	std::optional<int64_t> number;
	std::optional<std::string> owner; ///< self or opponent
};

struct NoticeComponent
{
	std::string kind; ///< resource, resource_per_day or flag
	std::optional<std::string> resource;
	std::optional<int32_t> amount;
	std::optional<std::string> owner;
};

struct Notice
{
	std::string displayMode; ///< auto, modal or info
	std::vector<NoticeToken> text;
	std::vector<NoticeComponent> components;
};

/// Either a notice or why the notification is unsupported.
struct NoticeParse
{
	std::optional<Notice> notice;
	std::string unsupported;
};

namespace notice_detail
{
constexpr std::array<const char *, 7> RESOURCES{"wood", "mercury", "ore", "sulfur", "crystal", "gems", "gold"};

/// The baseline scenario's authored victory and loss texts. MetaString keeps
/// them as raw strings, so each maps to a stable identifier. None holds a
/// placeholder, so the player-name replacement getVictoryLossMessage appends
/// changes nothing and is not a delivered token.
constexpr std::array<std::pair<const char *, const char *>, 4> AUTHORED{{
	{"All opponents have been defeated.", "heroes.baseline.all_opponents_defeated"},
	{"Your opponent won.", "heroes.baseline.opponent_won"},
	{"Seven days without a town have passed.", "heroes.baseline.seven_days_without_town"},
	{"Your opponent lost their last town.", "heroes.baseline.opponent_lost_last_town"},
}};

/// Hero names the scenario's heroes carry.
constexpr std::array<const char *, 2> HERO_NAMES{"hero.core.orrin.name", "hero.core.edric.name"};

/// "<prefix><n>" for n in [low, high], as TextIdentifier writes it.
inline std::optional<int> indexed(const std::string & text, const std::string & prefix, int low, int high)
{
	if(text.size() <= prefix.size() || text.compare(0, prefix.size(), prefix) != 0)
		return std::nullopt;
	const std::string digits = text.substr(prefix.size());
	if(digits.size() > 3 || (digits.size() > 1 && digits[0] == '0'))
		return std::nullopt;
	int value = 0;
	for(char digit : digits)
	{
		if(digit < '0' || digit > '9')
			return std::nullopt;
		value = value * 10 + (digit - '0');
	}
	if(value < low || value > high)
		return std::nullopt;
	return value;
}

/// CCreature::getNameSingularTextID for a core creature.
inline bool creatureSingular(const std::string & text)
{
	const std::string prefix = "creatures.core.";
	const std::string suffix = ".name.singular";
	if(text.size() <= prefix.size() + suffix.size() || text.compare(0, prefix.size(), prefix) != 0
		|| text.compare(text.size() - suffix.size(), suffix.size(), suffix) != 0)
		return false;
	const std::string identifier = text.substr(prefix.size(), text.size() - prefix.size() - suffix.size());
	if(identifier.empty() || identifier.size() > 64)
		return false;
	const auto letter = [](char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
	if(!letter(identifier[0]))
		return false;
	for(char c : identifier)
		if(!letter(c) && !(c >= '0' && c <= '9') && c != '_')
			return false;
	return true;
}

inline bool is(const NoticeOperation & operation, NoticeOperationKind kind, const std::string & text)
{
	return operation.kind == kind && operation.text == text && operation.number == 0;
}

inline bool textOperation(const NoticeOperation & operation, NoticeOperationKind kind)
{
	return operation.kind == kind && operation.number == 0;
}

inline NoticeToken text(const char * op, const std::string & textID)
{
	return {op, textID, std::nullopt, std::nullopt};
}

inline NoticeToken owned(const char * op, const std::string & owner)
{
	return {op, std::nullopt, std::nullopt, owner};
}

inline NoticeToken number(const char * op, int64_t value)
{
	return {op, std::nullopt, value, std::nullopt};
}

inline const char * display(NoticeDisplay mode)
{
	switch(mode)
	{
		case NoticeDisplay::AUTO:
			return "auto";
		case NoticeDisplay::MODAL:
			return "modal";
		case NoticeDisplay::INFO:
			return "info";
	}
	return "";
}

inline NoticeParse unsupported(const std::string & reason)
{
	return {std::nullopt, reason};
}

inline NoticeParse accepted(NoticeDisplay mode, std::vector<NoticeToken> tokens, std::vector<NoticeComponent> components = {})
{
	return {Notice{display(mode), std::move(tokens), std::move(components)}, {}};
}
}

/// Recognizes one notification delivered to the player whose color is self in
/// a game whose only other player is opponent. Player references become
/// relative owners; any other player, raw text, identifier, component or
/// operation makes the notification unsupported.
inline NoticeParse parseNotice(NoticeDisplay mode, const std::vector<NoticeOperation> & operations,
	const std::vector<NoticeComponentInput> & components, int self, int opponent)
{
	using namespace notice_detail;
	using Kind = NoticeOperationKind;
	using Component = NoticeComponentInput;

	const auto relative = [&](int color) -> std::optional<std::string>
	{
		if(color == self)
			return std::string("self");
		if(color == opponent)
			return std::string("opponent");
		return std::nullopt;
	};
	const auto resourceComponent = [&](Component::Kind kind, int resource) -> std::optional<int32_t>
	{
		if(components.size() != 1)
			return std::nullopt;
		const auto & component = components.front();
		if(component.kind != kind || component.subtype != Component::Subtype::RESOURCE || component.number != resource || !component.value)
			return std::nullopt;
		return component.value;
	};

	if(operations.empty())
		return unsupported("empty notice");

	const auto & first = operations.front();

	// CGResource::collectRes without a custom message.
	if(is(first, Kind::APPEND_LOCAL_STRING, "core.advevent.113"))
	{
		if(mode != NoticeDisplay::INFO || operations.size() != 2 || !textOperation(operations[1], Kind::REPLACE_TEXTID_STRING))
			return unsupported("resource pickup notice with an unaudited shape");
		const auto resource = indexed(operations[1].text, "core.restypes.", 0, 6);
		const auto amount = resource ? resourceComponent(Component::Kind::RESOURCE, *resource) : std::nullopt;
		if(!amount)
			return unsupported("resource pickup notice with an unaudited resource or component");
		return accepted(mode, {text("append_text", first.text), text("replace_text", operations[1].text)},
			{{"resource", std::string(RESOURCES[*resource]), *amount, std::nullopt}});
	}

	// CGMine::flagMine for a standard resource.
	if(textOperation(first, Kind::APPEND_TEXTID_STRING) && first.text.compare(0, 14, "core.mineevnt.") == 0)
	{
		const auto resource = indexed(first.text, "core.mineevnt.", 0, 6);
		const auto amount = resource ? resourceComponent(Component::Kind::RESOURCE_PER_DAY, *resource) : std::nullopt;
		if(mode != NoticeDisplay::AUTO || operations.size() != 1 || !amount)
			return unsupported("mine notice with an unaudited shape");
		return accepted(mode, {text("append_text", first.text)},
			{{"resource_per_day", std::string(RESOURCES[*resource]), *amount, std::nullopt}});
	}

	// CGameHandler::giveExperience at the level limit.
	if(is(first, Kind::APPEND_LOCAL_STRING, "core.genrltxt.1"))
	{
		if(mode != NoticeDisplay::MODAL || operations.size() != 2 || !textOperation(operations[1], Kind::REPLACE_TEXTID_STRING) || !components.empty())
			return unsupported("experience limit notice with an unaudited shape");
		bool known = false;
		for(const char * name : HERO_NAMES)
			known = known || operations[1].text == name;
		if(!known)
			return unsupported("experience limit notice for an unaudited hero name");
		return accepted(mode, {text("append_text", first.text), text("replace_text", operations[1].text)});
	}

	// CGameHandler::getVictoryLossMessage: the message, the subject player's
	// name and the subject player's flag.
	if(first.kind == Kind::APPEND_RAW_STRING || is(first, Kind::APPEND_TEXTID_STRING, "core.genrltxt.659")
		|| is(first, Kind::APPEND_TEXTID_STRING, "core.genrltxt.5") || is(first, Kind::APPEND_TEXTID_STRING, "core.genrltxt.7"))
	{
		if(mode != NoticeDisplay::MODAL || operations.size() != 2 || !textOperation(operations[1], Kind::REPLACE_TEXTID_STRING)
			|| components.size() != 1 || components[0].kind != Component::Kind::FLAG || components[0].subtype != Component::Subtype::PLAYER
			|| components[0].value)
			return unsupported("raw-text or victory notice with an unaudited shape");
		const auto color = indexed(operations[1].text, "vcmi.capitalColors.", 0, 7);
		const auto owner = color ? relative(*color) : std::nullopt;
		if(!owner || components[0].number != *color)
			return unsupported("victory or loss notice about an unaudited player");
		std::vector<NoticeToken> tokens;
		if(first.kind == Kind::APPEND_RAW_STRING)
		{
			if(first.number != 0)
				return unsupported("victory or loss notice with an unaudited shape");
			for(const auto & [authored, identifier] : AUTHORED)
				if(first.text == authored)
					tokens.push_back(text("append_text", identifier));
			if(tokens.empty())
				return unsupported("victory or loss notice with unaudited authored text");
		}
		else
		{
			tokens.push_back(text("append_text", first.text));
			tokens.push_back(owned("replace_text", *owner));
		}
		return accepted(mode, std::move(tokens), {{"flag", std::nullopt, std::nullopt, *owner}});
	}

	// NewTurnProcessor::createInfoWindow, delivered with NewTurn to every player.
	if(textOperation(first, Kind::APPEND_LOCAL_STRING) && first.text.compare(0, 14, "core.arraytxt.") == 0)
	{
		const auto week = indexed(first.text, "core.arraytxt.", 130, 135);
		if(!week || mode != NoticeDisplay::MODAL || !components.empty())
			return unsupported("new week notice with an unaudited shape");
		std::vector<NoticeToken> tokens{text("append_text", first.text)};
		const auto creature = [&](size_t index)
		{
			return textOperation(operations[index], Kind::REPLACE_TEXTID_STRING) && creatureSingular(operations[index].text);
		};
		bool valid = false;
		switch(*week)
		{
			case 130: // a new month, with its random name
			case 133: // a new week, with its random name
			{
				const auto low = *week == 130 ? 32 : 43;
				const auto high = *week == 130 ? 41 : 57;
				valid = operations.size() == 2 && textOperation(operations[1], Kind::REPLACE_LOCAL_STRING)
					&& indexed(operations[1].text, "core.arraytxt.", low, high).has_value();
				if(valid)
					tokens.push_back(text("replace_text", operations[1].text));
				break;
			}
			case 131: // double growth of one creature
			case 134: // bonus growth of one creature
				valid = operations.size() == 3 && creature(1) && creature(2) && operations[1].text == operations[2].text;
				if(valid)
				{
					tokens.push_back(text("replace_text", operations[1].text));
					tokens.push_back(text("replace_text", operations[2].text));
				}
				break;
			case 132: // plague
				valid = operations.size() == 1;
				break;
			case 135: // deity of fire: base creature twice, growth, upgrade, growth
				valid = operations.size() == 6 && creature(1) && creature(2) && operations[1].text == operations[2].text
					&& operations[3].kind == Kind::REPLACE_POSITIVE_NUMBER && operations[3].text.empty() && creature(4)
					&& operations[5].kind == Kind::REPLACE_POSITIVE_NUMBER && operations[5].text.empty()
					&& operations[3].number == operations[5].number;
				if(valid)
				{
					tokens.push_back(text("replace_text", operations[1].text));
					tokens.push_back(text("replace_text", operations[2].text));
					tokens.push_back(number("replace_positive_number", operations[3].number));
					tokens.push_back(text("replace_text", operations[4].text));
					tokens.push_back(number("replace_positive_number", operations[5].number));
				}
				break;
		}
		if(!valid)
			return unsupported("new week notice with an unaudited shape");
		return accepted(mode, std::move(tokens));
	}

	return unsupported("unaudited notice template");
}
}
