// Real native MetaString/Component parsing, without a game or callback.
#include "StdInc.h"
#include "Dialog.h"
#include "Notice.h"
#include "lib/GameLibrary.h"
#include "lib/CSkillHandler.h"
#include "lib/entities/artifact/CArtHandler.h"
#include "lib/spells/CSpellHandler.h"

#include <cstdlib>
#include <functional>
#include <iostream>

namespace
{
int checks = 0;
JsonNode payloads;
void expect(bool condition, const char * message)
{
	++checks;
	if(!condition) { std::cerr << message << '\n'; std::exit(1); }
}
template<typename Error> void reject(const std::function<void()> & function)
{
	try { function(); }
	catch(const Error &) { ++checks; return; }
	expect(false, "Unsupported native payload was accepted");
}
MetaString text(int id)
{
	MetaString result;
	result.appendLocalString(EMetaText::ADVOB_TXT, id);
	return result;
}
Component gold(int amount) { return Component(ComponentType::RESOURCE, GameResID(6), amount); }
Component skill(int id, int amount) { return Component(ComponentType::PRIM_SKILL, PrimarySkill(id), amount); }
JsonNode notice(const MetaString & value, const std::vector<Component> & components,
	EInfoWindowMode mode = EInfoWindowMode::AUTO, bool enabled = true,
	const std::set<std::string> & names = {"hero.core.orrin.name", "hero.core.edric.name"})
{
	auto result = explorer_rl::normalizeNotice(mode, value, components, PlayerColor(0), names, enabled);
	payloads["notices"].Vector().push_back(result);
	return result;
}
void dialogs()
{
	struct Case { int id; const char * family; std::vector<Component> components; bool selection, cancel; };
	const std::vector<Case> cases{
		{84, "guarded_mine", {}, false, true}, {187, "guarded_mine", {}, false, true},
		{86, "neutral_free_join", {}, false, true}, {90, "neutral_paid_join", {gold(2000)}, false, true},
		{91, "neutral_flee", {}, false, true},
		{146, "chest", {gold(1500), Component(ComponentType::EXPERIENCE, 1100)}, true, false},
		{0, "arena", {skill(0, 2), skill(1, 2)}, true, false},
		{71, "school_magic", {skill(2, 1), skill(3, 1)}, true, true},
		{158, "school_war", {skill(0, 1), skill(1, 1)}, true, true}};
	for(const auto & item : cases)
	{
		auto message = text(item.id);
		if(item.id == 90) { message.replaceNumber(1); message.replaceNumber(2000); }
		if(item.id == 86 || item.id == 90 || item.id == 91) message.replaceTextID("creatures.core.pikeman.name.plural");
		const auto parsed = explorer_rl::normalizeDialog(message, item.components, item.selection, item.cancel);
		auto panel = parsed.panel;
		panel["ref"].String() = "synthetic-dialog";
		for(size_t index = 0; index < panel["choices"].Vector().size(); ++index)
			panel["choices"][index]["ref"].String() = "synthetic-choice-" + std::to_string(index);
		payloads["dialogs"].Vector().push_back(std::move(panel));
		expect(parsed.panel["family"].String() == item.family, "Dialog family changed");
		expect(parsed.replies == (item.selection ? item.cancel ? std::vector<int>{1, 2, 0} : std::vector<int>{1, 2}
			: std::vector<int>{1, 0}), "Native answers changed");
		expect(parsed.panel.Struct().size() == 6 && parsed.panel["ref"].isNull(), "Normalizer leaked native identity");
		for(const auto & choice : parsed.panel["choices"].Vector()) expect(choice.Struct().size() == 2, "Choice leaked answer ID");
		reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(message, item.components, !item.selection, item.cancel); });
		reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(message, item.components, item.selection, !item.cancel); });
		auto extra = message; extra.replaceNumber(98);
		reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(extra, item.components, item.selection, item.cancel); });
		auto components = item.components; components.emplace_back(ComponentType::FLAG, PlayerColor(0));
		reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(message, components, item.selection, item.cancel); });
		if(!item.components.empty())
		{
			components = item.components; components[0].value = -1;
			reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(message, components, item.selection, item.cancel); });
		}
		JsonNode raw; message.jsonSerialize(raw);
		for(const char * pool : {"exactStrings", "localStrings", "stringsTextID", "numbers"})
		{
			auto data = raw;
			data[pool].Vector().push_back(std::string(pool) == "localStrings" || std::string(pool) == "numbers" ? JsonNode(int64_t(42)) : JsonNode("hidden"));
			MetaString forged; forged.jsonDeserialize(data);
			reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(forged, item.components, item.selection, item.cancel); });
		}
	}
	for(const char * name : {"creatures.hota.pikeman.name.plural", "creatures.core.pikeman.name.singular"})
	{
		auto paid = text(90); paid.replaceNumber(1); paid.replaceNumber(2000); paid.replaceTextID(name);
		reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(paid, {gold(2000)}, false, true); });
	}
	auto paid = text(90); paid.replaceNumber(1); paid.replaceNumber(2000); paid.replaceTextID("creatures.core.pikeman.name.plural");
	reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(paid, {gold(2001)}, false, true); });
	reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(text(146), {Component(ComponentType::RESOURCE, PlayerColor(6), 1000), Component(ComponentType::EXPERIENCE, 500)}, true, false); });
	MetaString wrong; wrong.appendLocalString(EMetaText::GENERAL_TXT, 20146);
	reject<explorer_rl::UnsupportedDialog>([&] { explorer_rl::normalizeDialog(wrong, {gold(1000), Component(ComponentType::EXPERIENCE, 500)}, true, false); });
}
void simpleNotices()
{
	const auto abandoned = text(85);
	expect(notice(abandoned, {})["text"].Vector().size() == 1, "Abandoned mine victory notice changed");
	reject<explorer_rl::UnsupportedNotice>([&] { notice(abandoned, {}, EInfoWindowMode::AUTO, false); });
	reject<explorer_rl::UnsupportedNotice>([&] { notice(abandoned, {}, EInfoWindowMode::MODAL); });
	reject<explorer_rl::UnsupportedNotice>([&] { notice(abandoned, {gold(1000)}); });
	auto extra = abandoned; extra.replaceRawString("unrendered");
	reject<explorer_rl::UnsupportedNotice>([&] { notice(extra, {}); });
	for(int id : {1, 72, 73, 159, 160})
	{
		auto message = text(id); message.replaceRawString("unrendered object");
		auto other = text(id); other.replaceRawString("different unrendered object");
		expect(notice(message, {}) == notice(other, {}), "Unused object name leaked");
		expect(notice(message, {})["text"].Vector().size() == 1, "Unused object replacement leaked");
		reject<explorer_rl::UnsupportedNotice>([&] { notice(message, {}, EInfoWindowMode::AUTO, false); });
		reject<explorer_rl::UnsupportedNotice>([&] { notice(message, {}, EInfoWindowMode::MODAL); });
		auto bad = message; bad.replaceNumber(5);
		reject<explorer_rl::UnsupportedNotice>([&] { notice(bad, {}); });
	}
	for(int id : {87, 29})
	{
		MetaString message; message.appendLocalString(id == 87 ? EMetaText::ADVOB_TXT : EMetaText::GENERAL_TXT, id);
		expect(notice(message, {}, EInfoWindowMode::MODAL)["text"].Vector().size() == 1, "Native refusal notice changed");
		reject<explorer_rl::UnsupportedNotice>([&] { notice(message, {}, EInfoWindowMode::MODAL, false); });
	}
}
void staticNotices()
{
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	LIBRARY->initializeLibrary();
	const auto artifact = ArtifactID(ArtifactID::decode("core:centaurAxe"));
	const auto spell = SpellID(SpellID::decode("core:magicArrow"));
	const auto secondary = SecondarySkill(SecondarySkill::SCHOLAR);
	const std::vector<Component> rewards{skill(0, 1), Component(ComponentType::SEC_SKILL, secondary, 2), Component(ComponentType::SPELL, spell)};
	for(const auto & reward : rewards)
	{
		auto message = text(115);
		if(reward.type == ComponentType::SEC_SKILL) message.replaceName(secondary);
		if(reward.type == ComponentType::SPELL) message.replaceName(spell);
		message.replaceName(secondary); message.replaceName(spell); message.replaceRawString("Scholar");
		const auto parsed = notice(message, {reward});
		expect(parsed["text"].Vector().size() == 1, "Scholar latent names leaked");
		expect(parsed["components"].Vector().size() == 1, "Scholar invented a hidden reward");
		reject<explorer_rl::UnsupportedNotice>([&] { notice(message, {reward}, EInfoWindowMode::AUTO, false); });
		auto malformed = message; malformed.replaceTextID("spell.core.haste.name");
		reject<explorer_rl::UnsupportedNotice>([&] { notice(malformed, {reward}); });
		if(reward.type == ComponentType::PRIM_SKILL)
		{
			auto changed = text(115); changed.replaceTextID("skill.core.logistics.name"); changed.replaceTextID("spell.core.haste.name"); changed.replaceRawString("different");
			expect(notice(changed, {reward}) == parsed, "Unrendered Scholar variables affected learner payload");
		}
	}
	auto chest = text(145); chest.replaceName(artifact); chest.replaceRawString("Treasure Chest");
	expect(notice(chest, {Component(ComponentType::ARTIFACT, artifact)})["text"].Vector().size() == 2, "Chest did not retain only displayed artifact name");
	reject<explorer_rl::UnsupportedNotice>([&] { notice(chest, {Component(ComponentType::SPELL_SCROLL, spell)}); });
	const std::vector<SpellID> learned{SpellID(10), SpellID(15), SpellID(20)};
	for(int branch : {0, 1, 2})
	{
		MetaString meeting; meeting.appendLocalString(EMetaText::GENERAL_TXT, 139); meeting.replaceTextID("hero.core.orrin.name");
		std::vector<Component> components{Component(ComponentType::SEC_SKILL, secondary, 3)};
		const auto section = [&](int prefix, int suffix, const std::vector<SpellID> & spells)
		{
			meeting.appendLocalString(EMetaText::GENERAL_TXT, prefix);
			for(size_t i = 0; i < spells.size(); ++i)
			{
				meeting.appendName(spells[i]); components.emplace_back(ComponentType::SPELL, spells[i]);
				if(i + 2 == spells.size()) meeting.appendLocalString(EMetaText::GENERAL_TXT, 141);
				else if(i + 1 < spells.size()) meeting.appendRawString(", ");
			}
			meeting.appendLocalString(EMetaText::GENERAL_TXT, suffix); meeting.replaceTextID("hero.core.edric.name");
		};
		if(branch != 1) section(140, 142, learned);
		if(branch == 2) meeting.appendLocalString(EMetaText::GENERAL_TXT, 141);
		if(branch != 0) section(147, 148, {SpellID(25)});
		const auto parsed = notice(meeting, components, EInfoWindowMode::MODAL);
		expect(parsed["components"].Vector().size() == components.size(), "Scholar lost ordered visible spells");
		reject<explorer_rl::UnsupportedNotice>([&] { notice(meeting, components, EInfoWindowMode::MODAL, false); });
		reject<explorer_rl::UnsupportedNotice>([&] { notice(meeting, components, EInfoWindowMode::MODAL, true, {"hero.core.orrin.name"}); });
		auto wrong = components; wrong.back() = Component(ComponentType::SPELL, SpellID(30));
		reject<explorer_rl::UnsupportedNotice>([&] { notice(meeting, wrong, EInfoWindowMode::MODAL); });
		auto extra = meeting; extra.appendRawString(", ");
		reject<explorer_rl::UnsupportedNotice>([&] { notice(extra, components, EInfoWindowMode::MODAL); });
	}
}
}
int main(int argc, char ** argv)
{
	payloads["dialogs"].Vector();
	payloads["notices"].Vector();
	if(argc == 2 && std::string(argv[1]) == "--static") staticNotices();
	else { dialogs(); simpleNotices(); }
	std::cout << "PAYLOADS_BEGIN\n" << payloads.toString() << "\nPAYLOADS_END\n";
	std::cout << "native adventure normalizers: " << checks << " checks passed\n";
}
