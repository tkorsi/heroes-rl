# Replay viewing modes and historical match inspector

These workflows retain the historical two-Castle, Blue Edric/random-policy match.
They do not claim generic support for the latest Teacher checkpoint or graphical
reconstruction through the VCMI renderer. Commands run from the repository root.

The user clarified the viewing contract on 25 October: keep the current
schematic style for now, with two explicitly selectable presentation modes of
the same logical simulation:

- **Schematic replay:** a readable map, objects, fog, actions and outcome. The
  first verified built-in-opponent example is available; its current viewer is
  scoped to that recorded scenario and labels window/army actions in summary.
  The viewed side's hero and army appear beneath the map; army slots follow
  recorded state changes, with pre-battle counts explicitly labelled during
  automatic combat and no active army shown after the hero leaves the map.
- **VCMI graphics replay:** reconstruct a retained match through the original
  game renderer and encode a video. This remains pending; a schematic video
  does not establish that the original graphical replay is implemented.

Both modes must consume recorded play rather than make the policies choose
again. The replay source comprises the initial state/map and configuration,
ordered commands from every controller (including automatic battles), the
required RNG state/events, and pinned engine/rules/data identities. Retain
adapter/window events as well so a future graphical viewer can reconstruct the
historical interface context. Check replay against state/event digests. Rendering
is optional, offline and independent of episode validity or training speed.

The performance direction is a reusable logical-core `reset`/`step` interface
over the existing VCMI rules. Measure loading/reset, state transitions,
pathfinding/combat, opponent decisions, policy inference, transport and validation
separately before replacing expensive components. Reusing resources in persistent
workers and reducing transport/copying are candidates, subject to equivalent
outcomes and player-private observations. Millisecond complete games are an
aspiration to test, not an achieved speed or a promise; report transition latency
and complete-game throughput with episode lengths and worker counts.

For the v7 built-in-opponent interface, `experiments/recording/render_builtin_match.py`
provides a separate schematic match viewer. It draws the full map and the learner's
fog view from a fresh native `vcmireplay --contextual ... --trace ...` pass over an
already verified capture. This is a map reconstruction, not footage of the game
window; window interactions are summarized and the viewed hero's army slots
update from packet-bound recorded snapshots. During automatic combat the panel
explicitly shows the army entering battle, not unrecorded tactical casualty counts.
The viewer checks the original verification, the fresh native report, and the
trace-generation receipt's file hashes. It keeps raw traces, video and provenance
under ignored local `runs/` and `videos/` directories. It does not start games or
update policy weights. Its optional Pillow dependency is pinned separately in
`dependencies/video-viewer.lock`; the gameplay learner dependencies are unchanged.

### Native Swift replay inspector

Both map panels now display recorded player knowledge: Red on the left and Blue
on the right. Unknown tiles are fully black. Drawing and hit-testing intersect
the selected player's visible IDs with objects still present in the current
native frame, so stale IDs cannot resurrect collected resources or defeated
heroes/armies. Object cards retain the selected player's perspective. Captured
mines display an upper-right owner-color bar; neutral mines have no bar.
The actual Swift projection passes disappearance and neutral/Red/Blue ownership
controls. The accepted Swift app was rebuilt and opened with both player views.
See [validation and game analysis](../validation/replay-player-knowledge-2026-10-30.json).

The completed checkpoint-1753 doubled-choice self-play evaluation can also be
opened as a native-state timeline. This is a separate exporter from the older
scenario-specific exporter below:

```sh
python3 tools/export_selfplay_inspector.py "$COMPLETED_ONE_GAME_RUN" --output "$FRESH_PRIVATE_OUTPUT"
python3 tools/build_replay_inspector.py --match "$FRESH_PRIVATE_OUTPUT/match.json" --output "$PRIVATE_APP_PATH"
```

The export verifies the passed evaluation and source capture hashes, reconstructs
the native map trace with a 120-second guarded deadline, and includes every
native input/state. Its private trace retains spectator facts; the viewer filters
both map panels to the corresponding player's recorded knowledge.
Policy-only UI choices, policy probabilities, hero inventories and army panels
are explicitly outside this export; absent army data is shown as unavailable.
All raw bundles, traces and app builds stay in durable private artifacts.

The 30 October export passed native reconstruction and timeline coverage checks
for 11,076 inputs and 11,077 states. The Swift release build passed and the app
was opened to “Self-play · checkpoint 1753” with 11,077 timeline events. The
first attempt replayed successfully but rejected differing clock measurements;
the corrected comparison excludes only `timing_seconds`, retaining all replay
counters, boundaries and verification facts. A focused regression checks that
changes to checked counts, boundary or unchecked data remain different. This
is offline replay; no new gameplay, policy inference or optimizer update runs.
See [compact validation](../validation/selfplay-swift-replay-2026-10-30.json).

`apps/replay-inspector` is an offline SwiftUI macOS application (macOS 14+).
It retains the schematic video's two map views and adds previous/next arrows,
a scrubber, searchable event selection, and three timeline filters: engine
endpoints plus decisions, learner decisions only, and every contextual record.
The selected match contains 85 map states, 13 learner decisions, and 1,809
contextual records (1,894 displayed entries including initial/native states).

The right inspector exposes unabridged event JSON, the last delivered learner
observation, recorded selected-action log probabilities/value/support, and source
archives including learner history, bridge, server/contextual tapes, rollout,
verification and map trace. Resources, movement points, open-window context and
seven army slots are summarized beside the maps. Click a map tile to inspect its
coordinates, readable object names and visible object IDs. Map objects use distinct
semantic icons (hero, town, mine, neutral army and resource subtype), with owner
colors on their borders. Co-located heroes and towns are offset so both remain
visible. Clicking an icon selects that exact object and displays a description
card below the maps. Resource quantities and starting inventories come from the
verified map and are explicitly marked as spectator data. The learner hero card
also displays the latest delivered army, movement and hero-window statistics,
with packet freshness. Enemy current inventories remain unavailable rather than
being inferred from starting quantities. Selected cards update with timeline
navigation and hide when the selected object disappears from the chosen view.
These are schematic symbols, not original game sprites. No network inference or training runs in the app.

Map snapshots advance only when the client reaches a completed native input's
packet endpoint. Intermediate events retain the last completed snapshot. Army
annotations refresh only after the corresponding panel record is delivered in
the contextual stream; they never look ahead to a later message at the same
packet position. Decision entries show the decision before execution. A closed
window supplies no new army data; automatic combat shows the pre-battle army,
not live casualties. An absent hero has no displayed active army. Recorded policy
numbers are not explanations of neural reasoning or distributions over every
alternative. Raw binary commands remain available even when only their command
type is decoded. The exporter deliberately retains the selected two-Castle,
blue-random-policy scenario checks rather than claiming universal replay support.

From the repository root, using Python with `dependencies/video-viewer.lock`:

```bash
python3 -m experiments.recording.export_inspector "$EPISODE" \
  --trace "$TRACE" --report "$NATIVE_REPORT" \
  --output runs/replay-inspector/match.json
python3 tools/validate_replay_inspector.py runs/replay-inspector/match.json "$EPISODE"
python3 tools/build_replay_inspector.py --match runs/replay-inspector/match.json
open "runs/replay-inspector/Heroes Replay.app"
```

The app bundles that exported match for immediate offline opening. Other exported
bundles can be opened with Command-O. Its verification badge reflects the exporter's
checks, not an independent cryptographic verification of arbitrary edited bundles
by the Swift UI. Source and compact validation are tracked; raw bundles and local
ad-hoc-signed `.app` builds remain ignored. The original VCMI renderer is still
outside this inspector's scope.

