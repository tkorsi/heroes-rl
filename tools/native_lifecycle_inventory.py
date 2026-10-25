"""Explicit immutable input inventory for benchmark_native_lifecycle only."""
from pathlib import Path
import hashlib
import os
import stat

SCHEMA = 'heroes-native-runtime-inventory-v1'
ROLES = {'data', 'configuration', 'mods', 'additional_libraries'}
# Values are identity selectors, never a dump of the environment or credentials.
SELECTORS = ('HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME',
             'XDG_DATA_DIRS', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'VCMI_USER_DATA_DIR')


def file_identity(path):
    before = path.stat()
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'Input changed while hashing: {path}')
    return {'sha256': checksum, 'bytes': after.st_size,
            'executable': bool(after.st_mode & 0o111)}


def runtime_inventory(spec):
    """Hash declared roots, retaining names, empty directories and file targets.

    Directory symlinks are refused rather than silently omitting their contents.
    List their canonical targets as explicit roots instead. File symlinks retain
    both the link text and resolved target. No claim about actual loader use.
    """
    if type(spec) is not dict or set(spec) != {'schema', *ROLES} or spec['schema'] != SCHEMA:
        raise ValueError('Runtime spec requires schema and data/configuration/mods/additional_libraries lists')
    errors, roots = [], {}
    for role in sorted(ROLES):
        values = spec[role]
        if type(values) is not list or any(type(p) is not str for p in values):
            errors.append(f'{role}: expected a list of absolute real paths')
            continue
        if role in ('data', 'configuration') and not values:
            errors.append(f'{role}: provide at least one explicit existing input path')
        for value in values:
            try:
                root = Path(value)
                if not root.is_absolute() or str(root.resolve(strict=True)) != value:
                    raise ValueError('use the absolute canonical real path (resolve symlinks first)')
                entries = {}
                def visit(path):
                    relative = str(path.relative_to(root))
                    mode = path.lstat().st_mode
                    if stat.S_ISLNK(mode):
                        target = path.resolve(strict=True)
                        if not target.is_file():
                            raise ValueError(f'{path}: directory symlink unsupported; declare its real target separately')
                        entries[relative] = {'kind': 'symlink', 'link': os.readlink(path),
                                             'target': str(target), **file_identity(target)}
                    elif stat.S_ISDIR(mode):
                        entries[relative] = {'kind': 'directory'}
                        for child in sorted(path.iterdir()):
                            visit(child)
                    elif stat.S_ISREG(mode):
                        entries[relative] = {'kind': 'file', **file_identity(path)}
                    else:
                        raise ValueError(f'{path}: only regular files/directories are supported')
                visit(root)
                if not any(row['kind'] != 'directory' for row in entries.values()):
                    raise ValueError('input tree contains no files')
                roots[f'{role}:{value}'] = entries
            except (OSError, ValueError, RuntimeError) as error:
                errors.append(f'{role}: {value}: {error}')
    return {'schema': SCHEMA, 'admitted': not errors, 'errors': errors, 'roots': roots,
            'selectors': {key: os.environ.get(key) for key in SELECTORS},
            'cwd': str(Path.cwd().resolve()),
            'coverage': 'Declared input bytes only; no proof of resource selection, completeness, dynamic library loads or gameplay readiness'}


def require_inventory(spec, expected=None, current=None):
    current = runtime_inventory(spec) if current is None else current
    if not current['admitted']:
        raise ValueError('Runtime prerequisites unavailable: ' + '; '.join(current['errors']))
    if expected is not None and current != expected:
        changes = []
        for root in sorted(set(current['roots']) | set(expected.get('roots', {}))):
            before = expected.get('roots', {}).get(root, {})
            after = current['roots'].get(root, {})
            for name in sorted(set(before) | set(after)):
                if before.get(name) != after.get(name):
                    changes.append(root + '/' + name)
        for key in ('selectors', 'cwd', 'schema', 'admitted', 'coverage'):
            if current.get(key) != expected.get(key):
                changes.append(key)
        raise ValueError('Declared runtime drift: ' + ', '.join(changes[:30]))
    return current
