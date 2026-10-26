"""Tests of the benchmark record tools: lossless episode compression and identity digests (no game)."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign_summary as summary  # noqa: E402
import compress_episodes as compress  # noqa: E402


def episode(root, frames):
    folder = root / 'run-x' / 'batches' / 'train-000266-seat0' / 'execution' / 'episode-000001'
    (folder / 'agents' / 'red').mkdir(parents=True)
    (folder / 'engine').mkdir()
    (folder / 'agents' / 'red' / 'history-v8.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
    (folder / 'engine' / 'replay.jsonl').write_text('{"start": 1}\n')
    return folder


FRAMES = [dict(seq=1, kind='view', day=1), dict(seq=2, kind='issued', command=7), dict(seq=3, kind='decision', choice=2),
          dict(seq=4, kind='event', event=dict(kind='turn_end'), day=1), dict(seq=5, kind='terminal', won=False)]


class DigestTests(unittest.TestCase):
    def test_every_observation_counts_and_bookkeeping_and_record_numbers_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = summary.game_digest(episode(Path(tmp) / 'a', FRAMES))
            renumbered = [dict(f, seq=f['seq'] + 10) for f in FRAMES]
            self.assertEqual(summary.game_digest(episode(Path(tmp) / 'b', renumbered)), base)
            other_issue = [dict(f, command=8) if f['kind'] == 'issued' else f for f in FRAMES]
            self.assertEqual(summary.game_digest(episode(Path(tmp) / 'c', other_issue)), base)
            reordered = [FRAMES[0], FRAMES[1], FRAMES[3], FRAMES[2], FRAMES[4]]
            self.assertNotEqual(summary.game_digest(episode(Path(tmp) / 'd', reordered)), base)

    def test_missing_digests_are_not_comparable(self):
        self.assertIsNone(summary.same([('a', None), ('a', None)]))
        self.assertIsNone(summary.same([None, None]))
        self.assertIsNone(summary.same(['x']))
        self.assertIs(summary.same([['a', 'b'], ('a', 'b')]), True)
        self.assertIs(summary.same(['x', 'y']), False)
        self.assertIsNone(summary.present(summary.game_digest, '/nonexistent/episode-000001'))


class CompressionTests(unittest.TestCase):
    def test_the_archive_replaces_the_folder_losslessly(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = episode(Path(tmp), FRAMES)
            before = (summary.game_digest(folder), summary.raw_digest(folder))
            compress.compress(folder)
            self.assertFalse(folder.exists())
            self.assertEqual((summary.game_digest(folder), summary.raw_digest(folder)), before)
            manifest = json.loads(folder.with_name('episode-000001.manifest.json').read_text())
            self.assertEqual(set(manifest['files']), {'agents/red/history-v8.jsonl', 'engine/replay.jsonl'})

    def test_an_interrupted_removal_is_finished_and_the_archive_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = episode(Path(tmp), FRAMES)
            before = summary.game_digest(folder)
            with mock.patch.object(compress.shutil, 'rmtree', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    compress.compress(folder)
            (folder / 'agents' / 'red' / 'history-v8.jsonl').unlink()  # removal stopped half way
            self.assertEqual(summary.game_digest(folder), before)  # read from the verified archive
            compress.compress(folder)
            self.assertFalse(folder.exists())
            self.assertEqual(summary.game_digest(folder), before)

    def test_an_unfinished_archive_is_replaced_and_a_mismatch_keeps_the_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = episode(Path(tmp), FRAMES)
            folder.with_name('episode-000001.tar.gz.partial').write_bytes(b'truncated')
            compress.compress(folder)
            self.assertFalse(folder.exists())
            other = episode(Path(tmp) / 'second', FRAMES)
            shutil.copy(folder.with_name('episode-000001.tar.gz'), other.with_name('episode-000001.tar.gz'))
            shutil.copy(folder.with_name('episode-000001.manifest.json'), other.with_name('episode-000001.manifest.json'))
            (other / 'engine' / 'replay.jsonl').write_text('{"start": 2}\n')
            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                compress.compress(other)
            self.assertTrue(other.exists())


if __name__ == '__main__':
    unittest.main()
