#include "StdInc.h"
#include "Notice.h"

#include "../../lib/GameLibrary.h"
#include "../../lib/CSkillHandler.h"
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
	size_t limit = 64;
	int field = 0;
	std::vector<std::string> raw;
	std::vector<std::pair<EMetaText, ui32>> local;
	std::vector<std::string> ids;
	std::vector<int> messages;
	std::vector<int64_t> numbers;

	template<typename Values> NativePools & operator&(Values & values)
	{
		if(values.size() > limit)
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

std::vector<Token> readTokens(const MetaString & text, size_t limit = 64)
{
	NativePools data;
	data.limit = limit;
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

const Entity * shownContent(const Component & component)
{
	if(!LIBRARY) unsupported("Missing static notice content library");
	const Entity * content = nullptr;
	if(component.type == ComponentType::SPELL)
	{
		const auto id = component.subType.as<SpellID>();
		if(component.subType != ComponentSubType(id) || component.value || id.getNum() < 0
			|| !LIBRARY->spellh || static_cast<size_t>(id.getNum()) >= LIBRARY->spellh->size())
			unsupported("Invalid displayed notice spell");
		content = LIBRARY->spells()->getById(id);
	}
	else if(component.type == ComponentType::SEC_SKILL)
	{
		const auto id = component.subType.as<SecondarySkill>();
		if(component.subType != ComponentSubType(id) || !component.value || *component.value < 1 || *component.value > 3
			|| id.getNum() < 0 || !LIBRARY->skillh || static_cast<size_t>(id.getNum()) >= LIBRARY->skillh->size())
			unsupported("Invalid displayed notice secondary skill");
		content = LIBRARY->skills()->getById(id);
	}
	else if(component.type == ComponentType::ARTIFACT)
	{
		trophyContentID(component); // exact type/range/core checks, no instance lookup
		content = LIBRARY->artifacts()->getById(component.subType.as<ArtifactID>());
	}
	else unsupported("Unsupported displayed notice content");
	static const std::regex key("core:[A-Za-z][A-Za-z0-9_]{0,63}");
	if(!content || !std::regex_match(content->getJsonKey(), key)) unsupported("Unsupported displayed notice content key");
	return content;
}

JsonNode shownComponent(const Component & component)
{
	JsonNode result;
	for(const char * key : {"resource", "amount", "owner", "content_id", "skill"}) result[key] = JsonNode();
	if(component.type == ComponentType::PRIM_SKILL)
	{
		const auto skill = component.subType.as<PrimarySkill>();
		if(component.subType != ComponentSubType(skill) || skill.getNum() < 0 || skill.getNum() > 3
			|| !component.value || *component.value != 1) unsupported("Invalid displayed primary skill reward");
		static const std::array<const char *, 4> names{"attack", "defence", "spellpower", "knowledge"};
		result["kind"].String() = "primary_skill";
		result["skill"].String() = names.at(skill.getNum());
		result["amount"].Integer() = *component.value;
	}
	else
	{
		result["content_id"].String() = shownContent(component)->getJsonKey();
		result["kind"].String() = component.type == ComponentType::SPELL ? "spell"
			: component.type == ComponentType::SEC_SKILL ? "secondary_skill" : "artifact";
		if(component.value) result["amount"].Integer() = *component.value;
	}
	return result;
}

bool nameToken(const Token & token, const char * prefix)
{
	const std::regex pattern(std::string(prefix) + "\\.core\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.name");
	return token.operation == REPLACE_TEXT && std::regex_match(token.text, pattern);
}

/// Only the v7 enumerated informational families. Returning null delegates to
/// the unchanged legacy allowlist. No world query or unused substitution is
/// exposed: static catalog lookup resolves only a delivered visible component.
JsonNode adventureNotice(EInfoWindowMode mode, const std::vector<Token> & tokens,
	const std::vector<Component> & components, PlayerColor recipient, const std::set<std::string> & ownNames)
{
	if(tokens.empty()) return {};
	std::vector<Token> shown;
	auto displayedComponents = components;
	bool supported = false;
	// An abandoned mine's native victory continuation precedes flagMine.
	if(mode == EInfoWindowMode::AUTO && components.empty() && tokens.size() == 1
		&& exact(tokens[0], APPEND_LOCAL, "core.advevent.85"))
	{
		shown = tokens;
		supported = true;
	}
	if(mode == EInfoWindowMode::MODAL && components.empty() && tokens.size() == 1
		&& (exact(tokens[0], APPEND_LOCAL, "core.advevent.87") || exact(tokens[0], APPEND_LOCAL, "core.genrltxt.29")))
	{
		shown = tokens;
		supported = true;
	}
	// initializeGuards appends one object-name replacement to these templates,
	// which have no placeholder. Validate its operation, then discard it.
	if(mode == EInfoWindowMode::AUTO && components.empty() && tokens.size() == 2 && tokens[1].operation == REPLACE_RAW)
		for(const int id : {1, 72, 73, 159, 160})
			if(exact(tokens[0], APPEND_LOCAL, "core.advevent." + std::to_string(id)))
			{
				shown = {tokens[0]};
				supported = true;
			}
	if(mode == EInfoWindowMode::AUTO && components.size() == 1 && tokens.size() == 3
		&& exact(tokens[0], APPEND_LOCAL, "core.advevent.145") && tokens[2].operation == REPLACE_RAW
		&& components[0].type == ComponentType::ARTIFACT
		&& exact(tokens[1], REPLACE_TEXT, shownContent(components[0])->getNameTextID()))
	{
		shown = {tokens[0], tokens[1]};
		supported = true;
	}
	// Scholar115 has zero placeholders. Its selected reward name and both
	// initialized latent variables are unrendered; retain just the actual
	// delivered primary/secondary/spell component and the template.
	if(mode == EInfoWindowMode::AUTO && components.size() == 1
		&& exact(tokens[0], APPEND_LOCAL, "core.advevent.115"))
	{
		const auto & component = components[0];
		// Reward::loadComponents uses an explicit zero for a learnable spell.
		// It is a native learnability marker, not a displayed numeric amount.
		// Only Scholar115 permits this form; meeting spells still require no value.
		if(component.type == ComponentType::SPELL)
		{
			if(!component.value || *component.value != 0)
				unsupported("Invalid displayed Scholar spell marker");
			displayedComponents[0].value.reset();
		}
		const auto & displayed = displayedComponents[0];
		const bool selectedName = component.type == ComponentType::SPELL || component.type == ComponentType::SEC_SKILL;
		const size_t latent = selectedName ? 2 : 1;
		if(component.type != ComponentType::PRIM_SKILL && !selectedName) unsupported("Invalid Scholar reward component");
		if(tokens.size() != latent + 3 || !nameToken(tokens[latent], "skill")
			|| !nameToken(tokens[latent + 1], "spell") || tokens[latent + 2].operation != REPLACE_RAW
			|| (selectedName && !exact(tokens[1], REPLACE_TEXT, shownContent(displayed)->getNameTextID())))
			unsupported("Unsupported Scholar reward grammar");
		if(selectedName && tokens[1].text != tokens[latent + (component.type == ComponentType::SPELL)].text)
			unsupported("Scholar selected reward differs from its native variable");
		shown = {tokens[0]};
		supported = true;
	}
	if(mode == EInfoWindowMode::MODAL && exact(tokens[0], APPEND_LOCAL, "core.genrltxt.139"))
	{
		if(tokens.size() < 6 || components.size() < 2 || components[0].type != ComponentType::SEC_SKILL
			|| components[0].subType != ComponentSubType(SecondarySkill(SecondarySkill::SCHOLAR)))
			unsupported("Unsupported meeting Scholar components");
		shownContent(components[0]);
		size_t position = 1, component = 1;
		const auto heroName = [&]()
		{
			if(position >= tokens.size() || !nameToken(tokens[position], "hero") || !ownNames.contains(tokens[position].text))
				unsupported("Meeting Scholar hero is not currently owned");
			return tokens[position++].text;
		};
		const auto first = heroName();
		std::string second;
		std::set<int> allSpells;
		const auto section = [&](int prefix, int suffix)
		{
			if(position >= tokens.size() || !exact(tokens[position], APPEND_LOCAL, "core.genrltxt." + std::to_string(prefix))) return false;
			++position;
			const size_t start = position;
			while(position < tokens.size() && !exact(tokens[position], APPEND_LOCAL, "core.genrltxt." + std::to_string(suffix))) ++position;
			if(position == tokens.size()) unsupported("Incomplete meeting Scholar section");
			const size_t end = position;
			const size_t count = (end - start + 1) / 2;
			if(!count || end - start != count * 2 - 1) unsupported("Invalid meeting Scholar list");
			int previous = -1;
			for(size_t i = 0; i < count; ++i)
			{
				if(component >= components.size() || components[component].type != ComponentType::SPELL)
					unsupported("Missing meeting Scholar spell component");
				const auto & spell = components[component++];
				const int id = spell.subType.as<SpellID>().getNum();
				if(id <= previous || !allSpells.insert(id).second
					|| !exact(tokens[start + 2 * i], APPEND_TEXT, shownContent(spell)->getNameTextID()))
					unsupported("Meeting Scholar spell order differs from components");
				previous = id;
				if(i + 1 < count && !(i + 2 == count
					? exact(tokens[start + 2 * i + 1], APPEND_LOCAL, "core.genrltxt.141")
					: exact(tokens[start + 2 * i + 1], APPEND_RAW, ", ")))
					unsupported("Unsupported meeting Scholar separator");
			}
			++position;
			const auto name = heroName();
			if(name == first || (!second.empty() && second != name)) unsupported("Meeting Scholar hero mismatch");
			second = name;
			return true;
		};
		const bool learns = section(140, 142);
		if(learns && position < tokens.size())
		{
			if(!exact(tokens[position++], APPEND_LOCAL, "core.genrltxt.141")) unsupported("Missing meeting Scholar conjunction");
			if(!section(147, 148)) unsupported("Missing meeting Scholar teaching section");
		}
		else if(!learns && !section(147, 148)) unsupported("Empty meeting Scholar notice");
		if(position != tokens.size() || component != components.size()) unsupported("Unused meeting Scholar fields");
		shown = tokens;
		supported = true;
	}
	if(!supported) return {};
	JsonNode result;
	result["display_mode"].String() = mode == EInfoWindowMode::AUTO ? "auto" : "modal";
	for(const auto & token : shown)
	{
		JsonNode value = token.operation == APPEND_RAW ? textToken("append_text", "heroes.notice.comma_space") : tokenJSON(token, recipient);
		for(const char * key : {"text_id", "number", "owner"}) if(value[key].isNull()) value[key] = JsonNode();
		result["text"].Vector().push_back(std::move(value));
	}
	result["components"].Vector();
	for(const auto & component : displayedComponents) result["components"].Vector().push_back(shownComponent(component));
	return result;
}
}

JsonNode normalizeNotice(EInfoWindowMode mode, const MetaString & text,
	const std::vector<Component> & components, PlayerColor recipient,
	const std::set<std::string> & ownHeroNameTextIds, bool adventureDialogs)
{
	if(components.size() > (adventureDialogs ? 128 : 32))
		unsupported("Too many informational notice components");
	// Validate recipient even for notices without a player replacement.
	relative(recipient.getNum(), recipient);
	const auto tokens = readTokens(text, adventureDialogs ? 512 : 64);
	if(tokens.empty())
		unsupported("Empty informational notice");
	if(adventureDialogs)
	{
		auto notice = adventureNotice(mode, tokens, components, recipient, ownHeroNameTextIds);
		if(!notice.isNull()) return notice;
	}
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
	// CGameHandler::giveHeroExperience: the recipient's current owned heroes,
	// including recruits. A syntactically valid name alone grants no visibility.
	static const std::regex heroName("hero\\.core\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.name");
	if(mode == EInfoWindowMode::MODAL && tokens.size() == 2 && components.empty()
		&& exact(tokens[0], APPEND_LOCAL, "core.genrltxt.1") && tokens[1].operation == REPLACE_TEXT
		&& std::regex_match(tokens[1].text, heroName)
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
