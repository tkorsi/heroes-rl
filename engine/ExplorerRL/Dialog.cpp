#include "StdInc.h"
#include "Dialog.h"

#include <array>
#include <limits>
#include <regex>
#include <type_traits>

namespace explorer_rl
{
namespace
{
[[noreturn]] void unsupported()
{
	throw UnsupportedDialog("Unsupported adventure dialog template or payload");
}

// Pinned MetaString serialization. Read every pool without localization or
// flattening local-family/index pairs, and reject all unused pool entries.
enum Operation { APPEND_RAW, APPEND_LOCAL, APPEND_TEXT, APPEND_NUMBER,
	REPLACE_RAW, REPLACE_LOCAL, REPLACE_TEXT, REPLACE_NUMBER, REPLACE_POSITIVE, NEWLINE };
struct Token { int operation; std::string text; int64_t number = 0; };
struct Pools
{
	int field = 0;
	std::vector<std::string> raw, ids;
	std::vector<std::pair<EMetaText, ui32>> local;
	std::vector<int> messages;
	std::vector<int64_t> numbers;
	template<typename Values> Pools & operator&(Values & values)
	{
		if(values.size() > 8) unsupported();
		if constexpr(std::is_same_v<Values, std::vector<std::string>>)
		{
			if(field == 0) raw = values;
			else if(field == 2) ids = values;
			else unsupported();
		}
		else if constexpr(std::is_same_v<Values, std::vector<std::pair<EMetaText, ui32>>>)
		{
			if(field != 1) unsupported();
			local = values;
		}
		else if constexpr(std::is_same_v<Values, std::vector<int64_t>>)
		{
			if(field != 4) unsupported();
			numbers = values;
		}
		else
		{
			static_assert(std::is_enum_v<typename Values::value_type>);
			if(field != 3) unsupported();
			for(const auto operation : values) messages.push_back(static_cast<int>(operation));
		}
		++field;
		return *this;
	}
};
template<typename T> const T & take(const std::vector<T> & values, size_t & index)
{
	if(index >= values.size()) unsupported();
	return values[index++];
}
std::vector<Token> read(const MetaString & text)
{
	Pools pools;
	auto copy = text;
	copy.serialize(pools);
	if(pools.field != 5 || !pools.raw.empty()) unsupported();
	size_t local = 0, ids = 0, numbers = 0;
	std::vector<Token> tokens;
	for(const auto operation : pools.messages)
	{
		Token token{operation, {}};
		if(operation == APPEND_LOCAL)
		{
			const auto & pair = take(pools.local, local);
			if(pair.first != EMetaText::ADVOB_TXT) unsupported();
			token.text = "core.advevent." + std::to_string(pair.second);
		}
		else if(operation == REPLACE_TEXT)
		{
			token.text = take(pools.ids, ids);
			if(token.text.size() > 128) unsupported();
		}
		else if(operation == REPLACE_NUMBER) token.number = take(pools.numbers, numbers);
		else unsupported();
		tokens.push_back(std::move(token));
	}
	if(local != pools.local.size() || ids != pools.ids.size() || numbers != pools.numbers.size()) unsupported();
	return tokens;
}
bool templateID(const Token & token, int index)
{
	return token.operation == APPEND_LOCAL && token.text == "core.advevent." + std::to_string(index);
}
bool creatureName(const Token & token, bool pluralOnly = false)
{
	static const std::regex any("creatures\\.core\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.name\\.(singular|plural)");
	static const std::regex plural("creatures\\.core\\.[A-Za-z][A-Za-z0-9_]{0,63}\\.name\\.plural");
	return token.operation == REPLACE_TEXT && std::regex_match(token.text, pluralOnly ? plural : any);
}
bool gold(const Component & component)
{
	return component.type == ComponentType::RESOURCE && component.subType == ComponentSubType(GameResID(GameResID::GOLD))
		&& component.value && *component.value > 0;
}
bool primary(const Component & component, int skill, int amount)
{
	return component.type == ComponentType::PRIM_SKILL && component.subType == ComponentSubType(PrimarySkill(skill))
		&& component.value && *component.value == amount;
}
JsonNode componentJSON(const Component & component)
{
	JsonNode result;
	for(const char * key : {"resource", "amount", "skill"}) result[key] = JsonNode();
	result["amount"].Integer() = *component.value;
	if(component.type == ComponentType::RESOURCE)
	{
		result["kind"].String() = "resource";
		result["resource"].String() = "gold";
	}
	else if(component.type == ComponentType::EXPERIENCE) result["kind"].String() = "experience";
	else
	{
		static const std::array<const char *, 4> names{"attack", "defence", "spellpower", "knowledge"};
		result["kind"].String() = "primary_skill";
		result["skill"].String() = names.at(component.subType.as<PrimarySkill>().getNum());
	}
	return result;
}
}

NormalizedDialog normalizeDialog(const MetaString & text, const std::vector<Component> & components,
	bool selection, bool cancel)
{
	const auto tokens = read(text);
	if(tokens.empty() || components.size() > 2) unsupported();
	std::string family;
	if(!selection && cancel)
	{
		if(tokens.size() == 1 && components.empty() && (templateID(tokens[0], 84) || templateID(tokens[0], 187)))
			family = "guarded_mine";
		if(tokens.size() == 2 && components.empty() && creatureName(tokens[1]))
		{
			if(templateID(tokens[0], 86)) family = "neutral_free_join";
			if(templateID(tokens[0], 91)) family = "neutral_flee";
		}
		if(tokens.size() == 4 && templateID(tokens[0], 90) && tokens[1].operation == REPLACE_NUMBER
			&& tokens[1].number > 0 && tokens[1].number <= std::numeric_limits<int32_t>::max()
			&& tokens[2].operation == REPLACE_NUMBER && creatureName(tokens[3], true)
			&& components.size() == 1 && gold(components[0]) && tokens[2].number == *components[0].value)
			family = "neutral_paid_join";
	}
	if(selection && tokens.size() == 1 && components.size() == 2)
	{
		if(!cancel && templateID(tokens[0], 146) && gold(components[0])
			&& components[1].type == ComponentType::EXPERIENCE && components[1].subType == ComponentSubType()
			&& components[1].value && *components[1].value > 0) family = "chest";
		if(!cancel && templateID(tokens[0], 0) && primary(components[0], 0, 2) && primary(components[1], 1, 2)) family = "arena";
		if(cancel && templateID(tokens[0], 71) && primary(components[0], 2, 1) && primary(components[1], 3, 1)) family = "school_magic";
		if(cancel && templateID(tokens[0], 158) && primary(components[0], 0, 1) && primary(components[1], 1, 1)) family = "school_war";
	}
	if(family.empty()) unsupported();
	NormalizedDialog result;
	auto & panel = result.panel;
	panel["family"].String() = family;
	panel["selection"].Bool() = selection;
	panel["cancel"].Bool() = cancel;
	for(const auto & token : tokens)
	{
		JsonNode value;
		for(const char * key : {"text_id", "number", "owner"}) value[key] = JsonNode();
		value["op"].String() = token.operation == APPEND_LOCAL ? "append_text" : token.operation == REPLACE_TEXT ? "replace_text" : "replace_number";
		if(token.operation == REPLACE_NUMBER) value["number"].Integer() = token.number;
		else value["text_id"].String() = token.text;
		panel["text"].Vector().push_back(std::move(value));
	}
	panel["components"].Vector();
	for(const auto & component : components) panel["components"].Vector().push_back(componentJSON(component));
	const auto choice = [&](const char * kind, int index, int reply)
	{
		JsonNode value;
		value["kind"].String() = kind;
		value["component_index"] = index < 0 ? JsonNode() : JsonNode(index);
		panel["choices"].Vector().push_back(std::move(value));
		result.replies.push_back(reply);
	};
	if(selection)
	{
		for(size_t i = 0; i < components.size(); ++i) choice("select_component", i, i + 1);
		if(cancel) choice("cancel", -1, 0);
	}
	else
	{
		choice(family == "neutral_flee" ? "pursue" : "accept", -1, 1);
		choice(family == "neutral_flee" ? "let_flee" : "decline", -1, 0);
	}
	return result;
}
}
