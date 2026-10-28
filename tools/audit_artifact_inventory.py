#!/usr/bin/env python3
"""Generate separate, read-only inventory instrumentation for captured replay.

This tool only writes source copies and their hash manifest. It does not build,
launch a game, replay a seed, alter an accepted runtime, or admit training data.
Compile the generated sources in place of the replay executable's two original
translation units. The resulting executable requires a fresh absolute private
HEROES_ARTIFACT_INVENTORY_AUDIT JSONL path. Its report uses a diagnostic schema
that the production training verifier does not accept.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "heroes-artifact-inventory-diagnostic-v1"


def replace_once(source, anchor, replacement):
    count = source.count(anchor)
    if count != 1:
        raise ValueError(f"Expected one diagnostic injection anchor, found {count}: {anchor!r}")
    return source.replace(anchor, replacement, 1)


def instrument_contextual(source):
    source = replace_once(source, '#include "StdInc.h"',
                          '#include "StdInc.h"\n#include "ArtifactInventoryAudit.h"')
    anchor = "\tGAME = std::make_unique<GameInstance>();"
    source = replace_once(source, anchor,
        "\tartifact_inventory_audit::Audit artifactAudit;\n\tartifactAudit.initialize(*state);\n" + anchor)
    source = replace_once(source, 'evidence->setAutomaticRequest([&client](const JsonNode & item)',
        'evidence->setAutomaticRequest([&client, &artifactAudit, &accounting](const JsonNode & item)')
    anchor = "\t\tclient.acceptExperimentReplayAutomaticRequest(*request);"
    source = replace_once(source, anchor,
        "\t\tartifactAudit.request(client.gameState(), *request, item, accounting);\n" + anchor)
    anchor = "\t\t\t\tclient.handlePack(*packet);"
    source = replace_once(source, anchor, anchor +
        "\n\t\t\t\tartifactAudit.afterPacket(client.gameState(), packetIndex, CTypeList::getInstance().getTypeID(packet));")
    anchor = '\t\tif(learner >= 0) result["request_accounting"] = accounting;'
    source = replace_once(source, anchor, anchor + "\n\t\tartifactAudit.complete();")
    return source


def instrument_main(source):
    # Change every assignment, including the later built-in schema selection.
    for old in ('"heroes-replay-verification-v1"', '"heroes-contextual-replay-verification-v1"',
                '"heroes-contextual-builtin-replay-verification-v1"'):
        source = replace_once(source, old, json.dumps(REPORT_SCHEMA))
    anchor = '\treport["log"].String() = logPath;'
    return replace_once(source, anchor,
        '\treport["diagnostic_only"].Bool() = true;\n'
        '\treport["training_admissible"].Bool() = false;\n' + anchor)


def absolute_includes(source, original_directory):
    def replace(match):
        name = match.group(1)
        candidate = original_directory / name
        if candidate.is_file():
            return '#include ' + json.dumps(str(candidate.resolve()))
        return match.group(0)
    return re.sub(r'^#include "([^"]+)"', replace, source, flags=re.MULTILINE)


def binding(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), size=len(data), sha256=hashlib.sha256(data).hexdigest())


def generate(output, root=ROOT):
    output = Path(output).resolve()
    root = Path(root).resolve()
    generated = root / 'vcmi/experimentreplay'
    originals = root / 'engine/ExperimentReplay'
    sources = {}
    for name, instrument in [('ContextualReplay.cpp', instrument_contextual), ('ReplayMain.cpp', instrument_main)]:
        original = originals / name
        if original.read_bytes() != (generated / name).read_bytes():
            raise ValueError(f"Canonical/generated source mismatch: {name}")
        sources[name] = absolute_includes(instrument(original.read_text()), generated)
    header = root / 'tools/native/ArtifactInventoryAudit.h'
    output.mkdir(parents=True, exist_ok=False)
    (output / header.name).write_bytes(header.read_bytes())
    for name, source in sources.items():
        (output / name).write_text(source)
    manifest = dict(schema='heroes-artifact-inventory-instrumentation-v1', diagnostic_only=True,
        training_admissible=False, source_bindings=[binding(originals / name) for name in sources],
        helper=binding(header), generator=binding(Path(__file__)),
        generated=[binding(output / name) for name in [*sources, header.name]],
        report_schema=REPORT_SCHEMA,
        limitations=['Inventory snapshots cover heroes on the map, not dismissed or tavern-pool heroes.',
            'Request inspection supports ordinary hero slots; other location kinds are explicitly unsupported.',
            'Slot compatibility is not full server authorization; requests are never executed by the audit.'])
    (output / 'instrumentation.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh private source-output directory')
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
