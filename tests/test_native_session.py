"""Native session contracts only; fixtures contain no Heroes Data or game rules."""
from pathlib import Path
import json
import socket
import threading
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def native_include_flags(engine):
    includes = [engine, engine / 'include', engine / 'build']
    if Path('/opt/homebrew/include/boost').exists():
        includes.append(Path('/opt/homebrew/include'))
    return ['-I' + str(path) for path in includes]


@unittest.skipUnless(shutil.which('g++') or shutil.which('clang++'), 'requires a C++20 compiler')
class NativeSessionChannelTests(unittest.TestCase):
    def test_production_channel_consecutive_generations_and_isolation(self):
        engine = ROOT / 'vcmi'
        library = engine / 'build/bin'
        if not any(library.glob('libvcmi.*')):
            self.skipTest('requires prepared and rebuilt VCMI shared library')
        compiler = shutil.which('g++') or shutil.which('clang++')
        with tempfile.TemporaryDirectory(prefix='native-session-channel-') as temporary:
            binary = Path(temporary) / 'channel-contract'
            command = [compiler, '-std=gnu++20', '-pthread', *native_include_flags(engine),
                       str(ROOT / 'tests/native/session_channel.cpp'), '-L' + str(library),
                       '-lvcmi', '-Wl,-rpath,' + str(library), '-o', str(binary)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('100 native channel generations', result.stdout)
            self.assertIn('no gameplay simulated', result.stdout)

    def test_legacy_socket_backend_framing_cancellation_and_limit(self):
        engine = ROOT / 'vcmi'
        library = engine / 'build/bin'
        if not any(library.glob('libvcmi.*')):
            self.skipTest('requires prepared and rebuilt VCMI shared library')
        compiler = shutil.which('g++') or shutil.which('clang++')
        with tempfile.TemporaryDirectory(prefix='native-socket-') as temporary:
            binary = Path(temporary) / 'socket-contract'
            command = [compiler, '-std=gnu++20', '-pthread', *native_include_flags(engine),
                       str(ROOT / 'tests/native/socket_policy.cpp'),
                       str(engine / 'AI/ExplorerRL/SocketPolicyTransport.cpp'),
                       '-I' + str(ROOT / 'engine/ExplorerRL'), '-L' + str(library),
                       '-lvcmi', '-Wl,-rpath,' + str(library), '-o', str(binary)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            for mode in ('normal', 'cancel', 'oversized'):
                with self.subTest(mode=mode), socket.socket() as listener:
                    listener.bind(('127.0.0.1', 0))
                    listener.listen()
                    listener.settimeout(3)
                    seen, errors = [], []
                    def peer():
                        try:
                            connection, _ = listener.accept()
                            with connection:
                                connection.settimeout(3)
                                line = b''
                                while not line.endswith(b'\n'):
                                    line += connection.recv(4096)
                                seen.append(json.loads(line))
                                if mode == 'normal':
                                    connection.sendall(b'{"fixture":')
                                    connection.sendall(b'1}\n{"fixture":2}\n')
                                elif mode == 'oversized':
                                    connection.sendall(b'x' * 65537 + b'\n')
                                else:
                                    connection.recv(1)
                        except Exception as error:
                            errors.append(error)
                    thread = threading.Thread(target=peer)
                    thread.start()
                    result = subprocess.run([str(binary), str(listener.getsockname()[1]), mode],
                                            capture_output=True, text=True, timeout=5)
                    thread.join(4)
                    self.assertFalse(thread.is_alive())
                    self.assertFalse(errors, errors)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(seen, [{'fixture': 'line one\nline two'}])

    def test_native_host_exclusivity_and_unloaded_state(self):
        binary = ROOT / 'vcmi/build/bin/native_session_contract'
        if not binary.exists():
            self.skipTest('build the dedicated native_session_contract target')
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('No game state or resources loaded', result.stdout)


if __name__ == '__main__':
    unittest.main()
