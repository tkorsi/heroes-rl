"""Strict raw-evidence inventory and extraction boundaries for cloud recovery."""
import hashlib
import io
from pathlib import Path
import tarfile
import time

import unittest
from contextlib import contextmanager
import tempfile

from tools import audit_cloud1335_publication as audit


@contextmanager
def raises(kind, match=None):
    with unittest.TestCase().assertRaisesRegex(kind, match or ".*"):
        yield



def cases(names, values):
    def decorate(function):
        function._cases = (names.split(','), values)
        return function
    return decorate


def entry(payload):
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
            'path': '/untrusted/historical/path'}


def archive(tmp_path, members):
    path = tmp_path / 'raw.part00000'
    with tarfile.open(path, 'w:gz') as stream:
        for name, kind, payload in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(payload)
                stream.addfile(member, io.BytesIO(payload))
            else:
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    member.linkname = payload
                stream.addfile(member)
    return path


def verify(path, inventory, **kwargs):
    return audit.verify_raw_archive(path, inventory, deadline=time.monotonic() + 10, **kwargs)


def test_exact_inventory_safe_regular_extraction_and_data_link_not_followed(tmp_path):
    relative = 'execution/episode-000001/profile/data/vcmi/Data'
    target = tmp_path / 'secret'
    target.write_text('must never be copied')
    links = {'batch/' + relative: str(target)}
    path = archive(tmp_path, [('batch', tarfile.DIRTYPE, ''),
        ('batch/execution/report.json', tarfile.REGTYPE, b'{}'),
        ('batch/empty', tarfile.REGTYPE, b''), ('batch/' + relative, tarfile.SYMTYPE, str(target))])
    output = tmp_path / 'extracted'
    result = verify(path, {'execution/report.json': entry(b'{}'), 'empty': entry(b'')},
                    extract_to=output, allowed_symlinks=links)
    assert result['regular_files_verified'] == 2
    assert result['regular_bytes_verified'] == 2
    assert result['symlinks_recorded_not_followed'] == links
    assert (output / 'execution/report.json').read_bytes() == b'{}'
    assert (output / 'empty').read_bytes() == b''
    assert not (output / relative).exists()
    assert not any(p.is_symlink() for p in output.rglob('*'))
    assert target.read_text() == 'must never be copied'


@cases('name', ['../escape', 'batch/../escape', '/batch/file', 'other/file',
                                  'batch//file', 'batch/./file', 'batch\\file', 'C:/file'])
def test_unsafe_member_paths_rejected_without_extracting(tmp_path, name):
    path = archive(tmp_path, [(name, tarfile.REGTYPE, b'x')])
    with raises(ValueError, match='Unsafe|outside'):
        verify(path, {'file': entry(b'x')}, extract_to=tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('.raw-audit-*'))


@cases('name', ['../escape', '/absolute', 'a/../b', 'a//b', 'a\\b'])
def test_unsafe_inventory_paths_rejected(tmp_path, name):
    path = archive(tmp_path, [('batch/file', tarfile.REGTYPE, b'x')])
    with raises(ValueError, match='Unsafe'):
        verify(path, {name: entry(b'x')})


@cases('kind', [tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE])
def test_hardlinks_and_special_members_rejected(tmp_path, kind):
    path = archive(tmp_path, [('batch/file', kind, '/etc/passwd')])
    with raises(ValueError, match='unsafe member'):
        verify(path, {'file': entry(b'x')})


def test_duplicate_member_rejected_and_staging_removed(tmp_path):
    path = archive(tmp_path, [('batch/file', tarfile.REGTYPE, b'x'),
                             ('batch/file', tarfile.REGTYPE, b'x')])
    with raises(ValueError, match='Duplicate'):
        verify(path, {'file': entry(b'x')}, extract_to=tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
    assert not list(tmp_path.glob('.raw-audit-*'))


@cases('payload,inventory,error', [
    (b'x', {'file': entry(b'y')}, 'digest differs'),
    (b'x', {'file': entry(b'xx')}, 'file or size'),
    (b'x', {'other': entry(b'x')}, 'file or size'),
    (b'x', {'file': entry(b'x'), 'missing': entry(b'z')}, 'complete raw inventory'),
])
def test_hash_size_extra_and_missing_members_rejected(tmp_path, payload, inventory, error):
    path = archive(tmp_path, [('batch/file', tarfile.REGTYPE, payload)])
    with raises(ValueError, match=error):
        verify(path, inventory)


def test_unknown_symlink_rejected(tmp_path):
    path = archive(tmp_path, [('batch/file', tarfile.SYMTYPE, '/etc/passwd')])
    with raises(ValueError, match='Unapproved'):
        verify(path, {'file': entry(b'x')})


def test_changed_or_missing_recorded_data_link_rejected(tmp_path):
    path = archive(tmp_path, [('batch/file', tarfile.REGTYPE, b'x')])
    with raises(ValueError, match='recorded game-data links'):
        verify(path, {'file': entry(b'x')}, allowed_symlinks={'batch/Data': '/recorded/Data'})
    path = archive(tmp_path, [('batch/Data', tarfile.SYMTYPE, '/wrong/Data')])
    with raises(ValueError, match='Unapproved'):
        verify(path, {'file': entry(b'x')}, allowed_symlinks={'batch/Data': '/recorded/Data'})


@cases('link_first', [True, False])
def test_archive_cannot_place_member_below_skipped_link(tmp_path, link_first):
    members = [('batch/Data', tarfile.SYMTYPE, '/recorded/Data'),
               ('batch/Data/secret', tarfile.REGTYPE, b'x')]
    path = archive(tmp_path, members if link_first else members[::-1])
    with raises(ValueError, match='below a symlink'):
        verify(path, {'Data/secret': entry(b'x')}, allowed_symlinks={'batch/Data': '/recorded/Data'})


def test_refuse_existing_output_or_symlink_parent(tmp_path):
    path = archive(tmp_path, [('batch/file', tarfile.REGTYPE, b'x')])
    existing = tmp_path / 'existing'
    existing.mkdir()
    with raises(ValueError, match='fresh'):
        verify(path, {'file': entry(b'x')}, extract_to=existing)
    linked = tmp_path / 'linked'
    linked.symlink_to(existing, target_is_directory=True)
    with raises(ValueError, match='symlink'):
        verify(path, {'file': entry(b'x')}, extract_to=linked / 'output')
    assert not (existing / 'output').exists()


def test_archive_input_symlink_and_expired_deadline_rejected(tmp_path):
    path = archive(tmp_path, [('batch/file', tarfile.REGTYPE, b'x')])
    link = tmp_path / 'link'
    link.symlink_to(path)
    with raises(ValueError, match='Symlink'):
        verify(link, {'file': entry(b'x')})
    with raises(TimeoutError):
        audit.verify_raw_archive(path, {'file': entry(b'x')}, deadline=time.monotonic() - 1)


def test_duplicate_json_keys_rejected():
    with raises(ValueError, match='Duplicate JSON'):
        audit.strict_json('{"entries":{"x":1,"x":2}}')


def test_full_state_fingerprint_changes_for_optimizer_or_sampler(tmp_path):
    import torch
    raw = {'model_state': {'w': torch.tensor([1.0])}, 'rng': {'torch': torch.tensor([4], dtype=torch.uint8)},
           'opponent_pool': [], 'resume': {'optimizer_state': {'step': torch.tensor(2.)},
           'progress': {'updates': 1}, 'sampler_state': {'next': 4}}}
    path = tmp_path / 'resume.pt'
    torch.save(raw, path)
    before = audit.full_state_fingerprints(path)
    raw['resume']['sampler_state']['next'] = 5
    torch.save(raw, path)
    after = audit.full_state_fingerprints(path)
    assert before['component_sha256']['model'] == after['component_sha256']['model']
    assert before['component_sha256']['sampler'] != after['component_sha256']['sampler']
    assert before['full_state_sha256'] != after['full_state_sha256']


class CloudPublicationAuditTests(unittest.TestCase):
    pass


def install_tests():
    for name, function in list(globals().items()):
        if not name.startswith('test_'):
            continue
        names, values = getattr(function, '_cases', ([], [()]))
        for index, value in enumerate(values):
            args = value if len(names) != 1 else (value,)
            def run(self, function=function, args=args):
                with tempfile.TemporaryDirectory() as directory:
                    if function.__code__.co_argcount:
                        function(Path(directory).resolve(), *args)
                    else:
                        function()
            setattr(CloudPublicationAuditTests, name + (f'_{index}' if names else ''), run)


install_tests()
