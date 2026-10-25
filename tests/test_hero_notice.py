"""Recruited-hero level-cap notices: native entitlement and v5 wire checks."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from experiments.learner import contextual as c


ROOT = Path(__file__).resolve().parents[1]


def cap_notice(name="hero.core.valeska.name"):
    return c.Notice("modal", (c.NoticeToken("append_text", text_id="core.genrltxt.1"),
                             c.NoticeToken("replace_text", text_id=name)), ())


def wire_notice(notice):
    return json.loads(json.dumps(asdict(notice)))


class HeroNoticeTests(unittest.TestCase):
    def test_recruited_names_roundtrip_in_v5_for_both_players(self):
        for player in (0, 1):
            for name in ("valeska", "catherine", "solmyr"):
                with self.subTest(player=player, name=name):
                    notice = cap_notice(f"hero.core.{name}.name")
                    frame = c.Frame("new-hero-notice", player, 0, 1, "event",
                                    event=c.VisibleEvent("notice", notice=notice), schema=c.HEROES_SCHEMA)
                    self.assertEqual(c.decode_frame(json.loads(json.dumps(c.encode_frame(frame)))), frame)

    def test_old_profiles_keep_recruited_name_notices_unsupported(self):
        for schema in (c.SCHEMA, c.TOWN_SCHEMA, c.ARMIES_SCHEMA):
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                c.decode_notice(wire_notice(cap_notice()), schema=schema)
            for name in ("orrin", "edric"):
                old = cap_notice(f"hero.core.{name}.name")
                self.assertEqual(c.decode_notice(wire_notice(old), schema=schema), old)

    def test_new_name_does_not_authorize_other_notice_templates(self):
        original = cap_notice()
        variants = (
            replace(original, display_mode="info"),
            replace(original, components=(c.NoticeComponent("flag", owner="self"),)),
            replace(original, text=tuple(reversed(original.text))),
            replace(original, text=(c.NoticeToken("append_text", text_id="core.genrltxt.5"), original.text[1])),
            replace(original, text=original.text + (c.NoticeToken("newline"),)),
            replace(original, text=(original.text[0], c.NoticeToken("append_text", text_id=original.text[1].text_id))),
        )
        for notice in variants:
            with self.subTest(notice=notice), self.assertRaises(ValueError):
                c.decode_notice(wire_notice(notice), schema=c.HEROES_SCHEMA)

    def test_malformed_or_noncore_name_is_rejected(self):
        for name in ("hero.hota.valeska.name", "hero.core.valeska.description",
                     "hero.core.9valeska.name", "hero.core.hidden.position.name",
                     "hero.core.valeska.name\n", "hero.core." + "x" * 65 + ".name"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cap_notice(name)

    def test_native_parser_requires_current_recipient_ownership(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        engine = ROOT / "vcmi"
        library = engine / "build/bin" / ("libvcmi.dylib" if sys.platform == "darwin" else "libvcmi.so")
        if compiler is None or not library.exists() or not (engine / "lib/texts/MetaString.h").exists():
            self.skipTest("Requires a compiler and the prepared, locally built real VCMI library")
        with tempfile.TemporaryDirectory(prefix="heroes-recruited-notice-") as directory:
            temporary = Path(directory)
            overlay = temporary / "AI/ExplorerRL"
            overlay.mkdir(parents=True)
            for name in ("Notice.h", "Notice.cpp", "StdInc.h"):
                shutil.copyfile(ROOT / "engine/ExplorerRL" / name, overlay / name)
            for name in ("lib", "Global.h"):
                (temporary / name).symlink_to(engine / name)
            fixture = temporary / "hero-notices.cpp"
            fixture.write_text(NATIVE_FIXTURE)
            executable = temporary / "hero-notices"
            includes = [overlay, engine, engine / "include", engine / "lib", engine / "build"]
            if Path("/opt/homebrew/include/boost").exists():
                includes.append(Path("/opt/homebrew/include"))
            command = [compiler, "-std=c++20", "-O1", "-Wall", "-Wextra", "-Werror",
                       "-Wno-unused-parameter", "-Wno-switch", "-Wno-sign-compare",
                       *(f"-I{path}" for path in includes), str(fixture), str(overlay / "Notice.cpp"),
                       f"-L{library.parent}", "-lvcmi", f"-Wl,-rpath,{library.parent}", "-o", str(executable)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
            native = json.loads(result.stdout)
            self.assertEqual(native["accepted"], 6)
            self.assertEqual(native["rejected"], 36)
            self.assertEqual(c.decode_notice(native["notice"], schema=c.HEROES_SCHEMA), cap_notice())


NATIVE_FIXTURE = r'''
// Real MetaString and canonical parser; no game, callback or fabricated engine.
#include "StdInc.h"
#include "Notice.h"
#include <cstdlib>
#include <functional>
#include <iostream>

namespace {
int accepted = 0, rejected = 0;
MetaString cap(const std::string & name) {
    MetaString text;
    text.appendLocalString(EMetaText::GENERAL_TXT, 1);
    text.replaceTextID(name);
    return text;
}
void reject(const std::function<void()> & operation) {
    try { operation(); }
    catch(const explorer_rl::UnsupportedNotice &) { ++rejected; return; }
    std::cerr << "Unexpectedly accepted ineligible hero notice\n";
    std::exit(1);
}
}
int main() {
    JsonNode output;
    for(int player : {0, 1}) {
        for(const std::string name : {"hero.core.valeska.name", "hero.core.catherine.name", "hero.core.solmyr.name"}) {
            const auto text = cap(name);
            const auto parse = [&](EInfoWindowMode mode, const MetaString & message,
                                   const std::vector<Component> & components,
                                   const std::set<std::string> & ownNames) {
                return explorer_rl::normalizeNotice(mode, message, components, PlayerColor(player), ownNames);
            };
            const auto notice = parse(EInfoWindowMode::MODAL, text, {}, {name});
            if(notice["text"][1]["text_id"].String() != name) return 2;
            ++accepted;
            if(name == "hero.core.valeska.name") output["notice"] = notice;
            // The same name is forbidden if unknown or visible but not owned;
            // an old observation of a formerly owned hero grants no entitlement.
            reject([&] { parse(EInfoWindowMode::MODAL, text, {}, {}); });
            reject([&] { parse(EInfoWindowMode::MODAL, text, {}, {"hero.core.orrin.name"}); });
            reject([&] { parse(EInfoWindowMode::INFO, text, {}, {name}); });
            reject([&] { parse(EInfoWindowMode::MODAL, text,
                               {Component(ComponentType::FLAG, PlayerColor(player))}, {name}); });
            auto extra = text;
            extra.replaceNumber(1);
            reject([&] { parse(EInfoWindowMode::MODAL, extra, {}, {name}); });
        }
        for(const std::string name : {"hero.hota.valeska.name", "hero.core.9valeska.name", "hero.core.hidden.position.name"})
            reject([&] { explorer_rl::normalizeNotice(EInfoWindowMode::MODAL, cap(name), {},
                                                       PlayerColor(player), {name}); });
    }
    output["accepted"].Integer() = accepted;
    output["rejected"].Integer() = rejected;
    std::cout << output.toString() << '\n';
}
'''


if __name__ == "__main__":
    unittest.main()
