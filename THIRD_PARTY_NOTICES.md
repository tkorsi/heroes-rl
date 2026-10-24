# Third-party notices

## VCMI

- Upstream: <https://github.com/vcmi/vcmi>
- Pinned revision: `6901839ae9760dfcd1d5ff2c4aa4ef6968fee752` (1.7.5)
- Copyright: VCMI Team and the contributors identified in the upstream source.
- Source-code license: GNU General Public License, version 2 or later.
- Upstream notice: [Copyright and license](https://github.com/vcmi/vcmi/blob/6901839ae9760dfcd1d5ff2c4aa4ef6968fee752/docs/readme.md#copyright-and-license).
- Verbatim license text: [VCMI-GPL-2.0.txt](LICENSES/VCMI-GPL-2.0.txt).

VCMI is obtained as an external dependency using `dependencies/vcmi.json` and
`tools/prepare_vcmi.py`. Its source checkout and compiled applications are not
part of this repository. Our changes are stored as patches under `engine/patches/`
and the external-agent integration under `engine/ExplorerRL/`. VCMI-derived
content retains its upstream license and attribution. These experiment changes
are not an upstream VCMI release.

VCMI's own dependencies and assets retain their respective upstream notices.
Preparing the dependency does not change their ownership or licenses.

## Heroes III and HotA

Original Heroes III game data, HotA installations, installers, artwork, sound,
music and other third-party game content are not redistributed here. Users must
provide their own compatible game installation locally. HotA is not needed for
the current baseline experiment.

The tracked `.vmap` archives contain authored JSON map definitions and asset
references only. They do not contain game artwork or other original game data.
Gameplay recordings and screenshots remain local outputs outside Git.

## Scope

These notices preserve third-party attribution and licensing. They do not select
a repository-wide license for independent project code, and they do not grant
rights to the original game assets.
