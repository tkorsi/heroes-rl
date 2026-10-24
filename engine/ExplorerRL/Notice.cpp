#include "StdInc.h"
#include "Notice.h"

#include "../../lib/GameLibrary.h"
#include "../../lib/entities/artifact/CArtHandler.h"
#include "../../lib/spells/CSpellHandler.h"

#include <array>
#include <regex>
#include <type_traits>

namespace explorer_rl
{
namespace
{
[[noreturn]] void unsupported(const char * reason)
{
	throw UnsupportedNotice(reason);
}

/// MetaString's private EMessage values, read through its public serializer.
/// Keep these in sync with the pinned engine; every operation and pool entry
/// must be consumed, including the one audited unused result substitution.
enum Operation { APPEND_RAW, APPEND_LOCAL, APPEND_TEXT, APPEND_NUMBER,
	REPLACE_RAW, REPLACE_LOCAL, REPLACE_TEXT, REPLACE_NUMBER, REPLACE_POSITIVE, NEWLINE };

struct Token
{
	int operation;
	std::string text;
	int64_t number = 0;
};

/// Read the public serialization fields without flattening local text pairs.
/// jsonSerialize encodes family*10000+index, which aliases malformed large
/// indices to another family; it cannot prove an exact native template.
struct NativePools
{
	int field = 0;
	std::vector<std::string> raw;
	std::vector<std::pair<EMetaText, ui32>> local;
	std::vector<std::string> ids;
	std::vector<int> messages;
	std::vector<int64_t> numbers;

	template<typename Values> NativePools & operator&(Values & values)
	{
		if(values.size() > 64)
			unsupported("Unsupported informational notice pool");
		if constexpr(std::is_same_v<Values, std::vector<std::string>>)
		{
			if(field == 0) raw = values;
			else if(field == 2) ids = values;
			else unsupported("Changed informational notice serialization");
		}
		else if constexpr(std::is_same_v<Values, std::vector<std::pair<EMetaText, ui32>>>)
		{
			if(field != 1) unsupported("Changed informational notice serialization");
			local = values;
		}
		else if constexpr(std::is_same_v<Values, std::vector<int64_t>>)
		{
			if(field != 4) unsupported("Changed informational notice serialization");
			numbers = values;
		}
		else
		{
			static_assert(std::is_enum_v<typename Values::value_type>);
			if(field != 3) unsupported("Changed informational notice serialization");
			for(const auto operation : values)
				messages.push_back(static_cast<int>(operation));
		}
		++field;
		return *this;
	}
};

template<typename Value> const Value & take(const std::vector<Value> & values, size_t & index)
{
	if(index >= values.size())
		unsupported("Incomplete informational notice pool");
	return values[index++];
}

std::string takeString(const std::vector<std::string> & values, size_t & index)
{
	const auto & value = take(values, index);
	if(value.size() > 256)
		unsupported("Unsupported informational notice text");
	return value;
}

std::string localID(const std::pair<EMetaText, ui32> & local)
{
	const auto [type, index] = local;
	if(type == EMetaText::GENERAL_TXT)
		return "core.genrltxt." + std::to_string(index);
	if(type == EMetaText::ARRAY_TXT)
		return "core.arraytxt." + std::to_string(index);
	if(type == EMetaText::ADVOB_TXT)
		return "core.advevent." + std::to_string(index);
	unsupported("Unsupported informational notice local text");
}

std::vector<Token> readTokens(const MetaString & text)
{
	NativePools data;
	auto copy = text; // native serialize is non-const; never mutate delivered text
	copy.serialize(data);
	if(data.field != 5)
		unsupported("Changed informational notice serialization");
	size_t li = 0, ri = 0, ti = 0, ni = 0;
	std::vector<Token> result;
	for(const auto operation : data.messages)
	{
		Token token{operation, {}};
		switch(operation)
		{
		case APPEND_RAW: case REPLACE_RAW:
			token.text = takeString(data.raw, ri);
			break;
		case APPEND_LOCAL: case REPLACE_LOCAL:
			token.text = localID(take(data.local, li));
			break;
		case APPEND_TEXT: case REPLACE_TEXT:
			token.text = takeString(data.ids, ti);
			break;
		case APPEND_NUMBER: case REPLACE_NUMBER: case REPLACE_POSITIVE:
			token.number = take(data.numbers, ni);
			break;
		case NEWLINE:
			break;
		default:
			unsupported("Unsupported informational notice operation");
		}
		result.push_back(std::move(token));
	}
	if(li != data.local.size() || ri != data.raw.size() || ti != data.ids.size() || ni != data.numbers.size())
		unsupported("Unused informational notice pool entry");
	return result;
}

bool exact(const Token & token, int operation, const std::string & text)
{
	return token.operation == operation && token.text == text;
}

bool localRange(const Token & token, int first, int last)
{
	for(int i = first; i <= last; ++i)
		if(exact(token, REPLACE_LOCAL, "core.arraytxt." + std::to_string(i)))
			return true;
	return false;
}

bool creature(const Token & token)
{
	// NewTurnProcessor::createInfoWindow supplies singular names, including
	// both base and upgraded creatures for Deity of Fire. These are announced
	// to all interfaces by the native new-week path, not inferred from a map.
	static const std::regex name("creatures\\.core\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.name\\.singular");
	return token.operation == REPLACE_TEXT && std::regex_match(token.text, name);
}

int playerName(const Token & token)
{
	if(exact(token, REPLACE_TEXT, "vcmi.capitalColors.0")) return 0;
	if(exact(token, REPLACE_TEXT, "vcmi.capitalColors.1")) return 1;
	return -1;
}

const char * relative(int player, PlayerColor recipient)
{
	if((player != 0 && player != 1) || (recipient.getNum() != 0 && recipient.getNum() != 1))
		unsupported("Unsupported informational notice player");
	return player == recipient.getNum() ? "self" : "opponent";
}

bool resourceComponent(const Component & component, ComponentType type, int resource)
{
	return component.type == type && component.value.has_value()
		&& component.subType == ComponentSubType(GameResID(resource));
}

bool flagComponent(const Component & component, int player)
{
	return component.type == ComponentType::FLAG && !component.value
		&& component.subType == ComponentSubType(PlayerColor(player));
}

std::string trophyContentID(const Component & component)
{
	// UIHelper::getArtifactsComponents supplies only the displayed artifact
	// type or scroll spell. Resolve that delivered ID in the static catalog;
	// never inspect an artifact instance, inventory, hero or game-state object.
	if(component.value || !LIBRARY)
		unsupported("Unsupported trophy component");
	const Entity * content = nullptr;
	if(component.type == ComponentType::ARTIFACT)
	{
		const auto id = component.subType.as<ArtifactID>();
		if(component.subType != ComponentSubType(id) || id.getNum() < 0 || id == ArtifactID::SPELL_SCROLL
			|| !LIBRARY->arth || static_cast<size_t>(id.getNum()) >= LIBRARY->arth->size())
			unsupported("Invalid trophy artifact type");
		content = LIBRARY->artifacts()->getById(id);
	}
	else if(component.type == ComponentType::SPELL_SCROLL)
	{
		const auto id = component.subType.as<SpellID>();
		if(component.subType != ComponentSubType(id) || id.getNum() < 0
			|| !LIBRARY->spellh || static_cast<size_t>(id.getNum()) >= LIBRARY->spellh->size())
			unsupported("Invalid trophy spell type");
		content = LIBRARY->spells()->getById(id);
	}
	else
		unsupported("Unsupported trophy component type");
	if(!content)
		unsupported("Missing trophy content type");
	const auto key = content->getJsonKey();
	static const std::regex coreKey("core:[A-Za-z][A-Za-z0-9_]{0,63}");
	if(!std::regex_match(key, coreKey))
		unsupported("Unsupported trophy content key");
	return key;
}

JsonNode textToken(const char * operation, const std::string & id)
{
	JsonNode result;
	result["op"].String() = operation;
	result["text_id"].String() = id;
	return result;
}

JsonNode tokenJSON(const Token & token, PlayerColor recipient)
{
	if(token.operation == APPEND_LOCAL || token.operation == APPEND_TEXT)
		return textToken("append_text", token.text);
	if(token.operation == REPLACE_LOCAL || token.operation == REPLACE_TEXT)
	{
		if(playerName(token) < 0)
			return textToken("replace_text", token.text);
		JsonNode result;
		result["op"].String() = "replace_text";
		result["owner"].String() = relative(playerName(token), recipient);
		return result;
	}
	if(token.operation == REPLACE_POSITIVE)
	{
		JsonNode result;
		result["op"].String() = "replace_positive_number";
		result["number"].Integer() = token.number;
		return result;
	}
	unsupported("Unsupported informational notice normalization");
}

std::string resultID(const Token & token)
{
	if(token.operation != APPEND_RAW)
		return {};
	// Exact literals authored by experiments/explorer/create_scenario.py.
	if(token.text == "All opponents have been defeated.") return "heroes.baseline.all_opponents_defeated";
	if(token.text == "Your opponent won.") return "heroes.baseline.opponent_won";
	if(token.text == "Seven days without a town have passed.") return "heroes.baseline.seven_days_without_town";
	if(token.text == "Your opponent lost their last town.") return "heroes.baseline.opponent_lost_last_town";
	return {};
}
}

JsonNode normalizeNotice(EInfoWindowMode mode, const MetaString & text,
	const std::vector<Component> & components, PlayerColor recipient,
	const std::set<std::string> & ownHeroNameTextIds)
{
	if(components.size() > 32)
		unsupported("Too many informational notice components");
	// Validate recipient even for notices without a player replacement.
	relative(recipient.getNum(), recipient);
	const auto tokens = readTokens(text);
	if(tokens.empty())
		unsupported("Empty informational notice");
	const char * displayMode = nullptr;
	switch(mode)
	{
	case EInfoWindowMode::AUTO: displayMode = "auto"; break;
	case EInfoWindowMode::MODAL: displayMode = "modal"; break;
	case EInfoWindowMode::INFO: displayMode = "info"; break;
	default: unsupported("Unsupported informational notice display mode");
	}

	bool supported = false;
	bool dropPlayerReplacement = false;
	std::string normalizedResult;
	std::vector<std::string> trophyContent;
	// CGResource::collectRes: the quantity is in the component, not text.
	if(mode == EInfoWindowMode::INFO && tokens.size() == 2 && components.size() == 1
		&& exact(tokens[0], APPEND_LOCAL, "core.advevent.113"))
		for(int resource = 0; resource < 7; ++resource)
			if(exact(tokens[1], REPLACE_TEXT, "core.restypes." + std::to_string(resource))
				&& resourceComponent(components[0], ComponentType::RESOURCE, resource))
				supported = true;
	// CGMine::flagMine: the resource identifier must agree with its component.
	if(mode == EInfoWindowMode::AUTO && tokens.size() == 1 && components.size() == 1)
		for(int resource = 0; resource < 7; ++resource)
			if(exact(tokens[0], APPEND_TEXT, "core.mineevnt." + std::to_string(resource))
				&& resourceComponent(components[0], ComponentType::RESOURCE_PER_DAY, resource))
				supported = true;
	// CGameHandler::giveHeroExperience: only the recipient's baseline hero.
	if(mode == EInfoWindowMode::MODAL && tokens.size() == 2 && components.empty()
		&& exact(tokens[0], APPEND_LOCAL, "core.genrltxt.1") && tokens[1].operation == REPLACE_TEXT
		&& (tokens[1].text == "hero.core.orrin.name" || tokens[1].text == "hero.core.edric.name")
		&& ownHeroNameTextIds.contains(tokens[1].text))
		supported = true;
	// visitBattleResultsApplied sends this exact zero-choice modal notice only
	// to the victor. Its ordered components are the trophies displayed by the
	// native UI, including each scroll's displayed spell, not hidden inventory.
	if(mode == EInfoWindowMode::MODAL && tokens.size() == 1 && !components.empty()
		&& exact(tokens[0], APPEND_LOCAL, "core.genrltxt.30"))
	{
		for(const auto & component : components)
			trophyContent.push_back(trophyContentID(component));
		supported = true;
	}
	// NewTurnProcessor::createInfoWindow; only this precise public announcement
	// grammar may carry creature names. Numeric growth may be signed natively.
	if(mode == EInfoWindowMode::MODAL && components.empty())
	{
		if(tokens.size() == 1 && exact(tokens[0], APPEND_LOCAL, "core.arraytxt.132"))
			supported = true;
		if(tokens.size() == 2 && ((exact(tokens[0], APPEND_LOCAL, "core.arraytxt.130") && localRange(tokens[1], 32, 41))
			|| (exact(tokens[0], APPEND_LOCAL, "core.arraytxt.133") && localRange(tokens[1], 43, 57))))
			supported = true;
		if(tokens.size() == 3 && (exact(tokens[0], APPEND_LOCAL, "core.arraytxt.131")
			|| exact(tokens[0], APPEND_LOCAL, "core.arraytxt.134"))
			&& creature(tokens[1]) && creature(tokens[2]) && tokens[1].text == tokens[2].text)
			supported = true;
		if(tokens.size() == 6 && exact(tokens[0], APPEND_LOCAL, "core.arraytxt.135")
			&& creature(tokens[1]) && creature(tokens[2]) && tokens[1].text == tokens[2].text
			&& tokens[3].operation == REPLACE_POSITIVE && creature(tokens[4])
			&& tokens[5].operation == REPLACE_POSITIVE && tokens[3].number == tokens[5].number)
			supported = true;
	}
	// CGameHandler::getVictoryLossMessage always appends exactly one player
	// replacement and flag. The authored raw messages and genrltxt.659 contain
	// no placeholder; omit only that exact, validated unused replacement.
	if(mode == EInfoWindowMode::MODAL && tokens.size() == 2 && components.size() == 1
		&& playerName(tokens[1]) >= 0 && flagComponent(components[0], playerName(tokens[1])))
	{
		normalizedResult = resultID(tokens[0]);
		if(!normalizedResult.empty() || exact(tokens[0], APPEND_TEXT, "core.genrltxt.659"))
		{
			supported = true;
			dropPlayerReplacement = true;
		}
		if(exact(tokens[0], APPEND_TEXT, "core.genrltxt.5") || exact(tokens[0], APPEND_TEXT, "core.genrltxt.7")
			|| exact(tokens[0], APPEND_TEXT, "core.genrltxt.8"))
			supported = true;
	}
	if(!supported)
		unsupported("Unsupported informational notice template");

	JsonNode result;
	result["display_mode"].String() = displayMode;
	auto & output = result["text"].Vector();
	for(size_t index = 0; index < tokens.size(); ++index)
	{
		if(index == 1 && dropPlayerReplacement)
			continue;
		output.push_back(index == 0 && !normalizedResult.empty()
			? textToken("append_text", normalizedResult) : tokenJSON(tokens[index], recipient));
	}
	static const std::array<const char *, 7> resources{"wood", "mercury", "ore", "sulfur", "crystal", "gems", "gold"};
	auto & outputComponents = result["components"].Vector();
	for(size_t index = 0; index < components.size(); ++index)
	{
		const auto & component = components[index];
		JsonNode value;
		if(!trophyContent.empty())
		{
			value["kind"].String() = component.type == ComponentType::ARTIFACT ? "artifact" : "spell_scroll";
			value["content_id"].String() = trophyContent[index];
		}
		else if(component.type == ComponentType::FLAG)
		{
			value["kind"].String() = "flag";
			value["owner"].String() = relative(component.subType.as<PlayerColor>().getNum(), recipient);
		}
		else
		{
			value["kind"].String() = component.type == ComponentType::RESOURCE ? "resource" : "resource_per_day";
			value["resource"].String() = resources.at(component.subType.as<GameResID>().getNum());
			value["amount"].Integer() = *component.value;
		}
		outputComponents.push_back(std::move(value));
	}
	return result;
}
}
