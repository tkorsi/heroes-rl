"""Diagnostic instrumentation must fail closed and stay outside admission."""
from pathlib import Path
import tempfile
import unittest

from tools.audit_artifact_inventory import (ROOT, REPORT_SCHEMA, absolute_includes,
    instrument_contextual, instrument_main, replace_once)


class ArtifactInventoryAuditTests(unittest.TestCase):
    def test_missing_or_duplicate_anchor_is_rejected(self):
        for source in ('missing', 'anchor anchor'):
            with self.assertRaises(ValueError):
                replace_once(source, 'anchor', 'replacement')

    def test_request_snapshot_precedes_delivery_and_packets_follow_application(self):
        source = (ROOT / 'engine/ExperimentReplay/ContextualReplay.cpp').read_text()
        result = instrument_contextual(source)
        self.assertLess(result.index('artifactAudit.request('),
                        result.index('client.acceptExperimentReplayAutomaticRequest(*request)'))
        self.assertLess(result.index('client.handlePack(*packet)'), result.index('artifactAudit.afterPacket('))
        self.assertIn('contextualRequestAccounting(tape, inputs, terminalInput, prefix, learner,', result)
        self.assertIn('artifactAudit.initialize(*state)', result)
        self.assertNotIn('moveArtifact(', result)

    def test_diagnostic_report_cannot_claim_production_schema(self):
        source = (ROOT / 'engine/ExperimentReplay/ReplayMain.cpp').read_text()
        result = instrument_main(source)
        self.assertNotIn('"heroes-contextual-builtin-replay-verification-v1"', result)
        self.assertNotIn('"heroes-contextual-replay-verification-v1"', result)
        self.assertNotIn('"heroes-replay-verification-v1"', result)
        self.assertEqual(result.count(REPORT_SCHEMA), 3)
        self.assertIn('report["training_admissible"].Bool() = false;', result)
        self.assertIn('replay.run(records, allowPlatformMismatch, prefix,', result)

    def test_relative_includes_resolve_only_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'Present.h').write_text('')
            result = absolute_includes('#include "Present.h"\n#include "Missing.h"\n', directory)
            self.assertIn(str(directory / 'Present.h'), result)
            self.assertIn('#include "Missing.h"', result)


if __name__ == '__main__':
    unittest.main()
