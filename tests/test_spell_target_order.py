"""Native spell-target regression; no game, replay or model is launched."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.test_builtin_native_contract import function

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "vcmi"


@unittest.skipUnless((ENGINE / "lib/spells/effects/UnitEffect.cpp").is_file(), "Prepared pinned source required")
class SpellTargetOrderTests(unittest.TestCase):
    def test_production_target_selection_is_independent_of_allocation_and_duplicates(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("A C++20 compiler is required")
        selected = (
            ("lib/spells/effects/UnitEffect.cpp", "EffectTarget UnitEffect::transformTargetByRange("),
            ("lib/spells/BattleSpellMechanics.cpp", "battle::Units BattleSpellMechanics::collectTargets() const"),
            ("lib/spells/BattleSpellMechanics.cpp", "std::vector<const CStack *> BattleSpellMechanics::getAffectedStacks("),
            ("lib/spells/effects/Damage.cpp", "int64_t Damage::damageForTarget("),
            ("lib/spells/effects/Damage.cpp", "void Damage::apply("),
        )
        production = "\n\n".join(function((ENGINE / path).read_text(), signature)
                                 for path, signature in selected)
        fixture = (ROOT / "tests/native/spell_target_order.cpp").read_text()
        self.assertEqual(fixture.count("// PRODUCTION_FUNCTIONS"), 1)
        program = fixture.replace("// PRODUCTION_FUNCTIONS", production)
        with tempfile.TemporaryDirectory(prefix="heroes-spell-target-order-") as temporary:
            source, binary = Path(temporary) / "regression.cpp", Path(temporary) / "regression"
            source.write_text(program)
            compiled = subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-pedantic",
                str(source), "-o", str(binary)], capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(checked.stdout, "576 allocation/input permutations verified\n")
