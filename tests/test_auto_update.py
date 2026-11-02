"""Реальные bash-функции обновления с шумным загрузчиком настроек."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


BASH = shutil.which("bash")
if BASH is None and sys.platform == "win32":
    candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    if candidate.exists():
        BASH = str(candidate)
SCRIPT = Path(__file__).resolve().parents[1] / "infra/alisa-update.sh"


@unittest.skipUnless(BASH, "Нужен bash (на Windows подходит Git Bash)")
class AutoUpdateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        package = self.root / "assistant"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "config.py").write_text(
            "import os\nfrom types import SimpleNamespace as NS\n"
            "def load_config():\n"
            "    print('Loaded config.local.yaml: server, reminders')\n"
            "    if os.environ.get('UPDATE_TEST_BAD_CONFIG'):\n"
            "        raise ValueError('invalid local config')\n"
            "    return NS(server=NS(port=int(os.environ['UPDATE_TEST_PORT']), token='test'))\n",
            encoding="utf-8",
        )
        self.status = {"working": 0, "idle_s": 300, "drafts": 0}
        self.http_code = 200
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/status" or self.headers.get("X-Token") != "test":
                    self.send_error(403)
                    return
                body = json.dumps(fixture.status).encode()
                self.send_response(fixture.http_code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def close():
            self.server.shutdown()
            self.server.server_close()
            thread.join()

        self.addCleanup(close)

    def run_function(self, name, **overrides):
        script = SCRIPT.read_text(encoding="utf-8")
        start = script.index(f"\n{name}() {{\n") + 1
        end = script.index("\n}\n", start) + 3
        program = (
            "set -euo pipefail\n"
            f"ROOT={shlex.quote(self.root.as_posix())}\n"
            f"PY={shlex.quote(Path(sys.executable).as_posix())}\n"
            "QUIET_S=120\nHEALTH_WAIT_S=5\nSETTLE_S=0\n"
            "SERVICES='alisa-reminders alisa-server alisa-telegram'\n"
            "declare -A RESTARTS_BEFORE=()\n"
            "as_owner() { \"$@\"; }\n"
            "log() { printf '%s\\n' \"$*\" >&2; }\n"
            "sleep() { :; }\n"
            # Проверяем точный URL, который получила бы настоящая curl.
            "curl() { [[ \"${@: -1}\" == \"http://127.0.0.1:$UPDATE_TEST_PORT/health\" ]]; }\n"
            "systemctl() { case \"$*\" in *NRestarts*) echo 0;; *ActiveState*) echo active;; "
            "*) return 1;; esac; }\n"
            + script[start:end] + f"\n{name}\n"
        )
        env = {**os.environ, "UPDATE_TEST_PORT": str(self.server.server_port),
               "PYTHONIOENCODING": "utf-8", **overrides}
        return subprocess.run([BASH, "-c", program], cwd=self.root, env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=15)

    def test_quiet_status_is_one_word_despite_local_config_output(self):
        result = self.run_function("server_state")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "quiet\n")

    def test_busy_status_is_preserved(self):
        for status in ({"working": 1, "idle_s": 300, "drafts": 0},
                       {"working": 0, "idle_s": 30, "drafts": 0},
                       {"working": 0, "idle_s": 300, "drafts": 1}):
            with self.subTest(status=status):
                self.status = status
                result = self.run_function("server_state")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "busy\n")

    def test_http_error_stays_unknown(self):
        self.http_code = 401
        self.assertEqual(self.run_function("server_state").stdout, "unknown\n")

    def test_invalid_config_cannot_report_quiet(self):
        result = self.run_function("server_state", UPDATE_TEST_BAD_CONFIG="1")
        self.assertEqual(result.stdout, "unknown\n")

    def test_health_url_contains_only_configured_port(self):
        result = self.run_function("healthy")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Loaded config.local.yaml", result.stderr)

    def test_health_check_fails_on_invalid_config(self):
        result = self.run_function("healthy", UPDATE_TEST_BAD_CONFIG="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
