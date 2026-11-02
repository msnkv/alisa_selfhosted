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

    def run_function(self, name, restart_first=False, **overrides):
        script = SCRIPT.read_text(encoding="utf-8")
        functions = []
        for function in (["restart_all", name] if restart_first else [name]):
            start = script.index(f"\n{function}() {{\n") + 1
            end = script.index("\n}\n", start) + 3
            functions.append(script[start:end])
        program = (
            "set -euo pipefail\n"
            f"ROOT={shlex.quote(self.root.as_posix())}\n"
            f"PY={shlex.quote(Path(sys.executable).as_posix())}\n"
            "QUIET_S=120\nHEALTH_WAIT_S=5\nSETTLE_S=0\n"
            "SERVICES='alisa-reminders alisa-server alisa-telegram'\n"
            "printf '%s\\n' \"${UPDATE_TEST_RESTARTS_BEFORE:-0}\" > \"$ROOT/restarts\"\n"
            "as_owner() { \"$@\"; }\n"
            "log() { printf '%s\\n' \"$*\" >&2; }\n"
            "sleep() { :; }\n"
            # Проверяем точный URL, который получила бы настоящая curl.
            "curl() { [[ \"${@: -1}\" == \"http://127.0.0.1:$UPDATE_TEST_PORT/health\" ]]; }\n"
            "systemctl() {\n"
            "  case \"$1\" in\n"
            "    restart) printf '%s\\n' \"${UPDATE_TEST_RESTARTS_AFTER:-0}\" > \"$ROOT/restarts\";;\n"
            "    show)\n"
            "      if [[ \"${@: -1}\" != alisa-telegram ]]; then\n"
            "        case \"$*\" in *NRestarts*) echo 0;; *ActiveState*) echo active;; esac\n"
            "      else\n"
            "        case \"$*\" in\n"
            "          *NRestarts*) cat \"$ROOT/restarts\";;\n"
            "          *ActiveState*) echo \"${UPDATE_TEST_SERVICE_STATE:-active}\";;\n"
            "          *) return 1;;\n"
            "        esac\n"
            "      fi;;\n"
            "    *) return 1;;\n"
            "  esac\n"
            "}\n"
            + "\n".join(functions)
            + ("\nrestart_all" if restart_first else "") + f"\n{name}\n"
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

    def test_planned_restart_clears_old_failures_without_rollback(self):
        result = self.run_function("healthy", restart_first=True,
                                   UPDATE_TEST_RESTARTS_BEFORE="1", UPDATE_TEST_RESTARTS_AFTER="0")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_new_crash_is_rejected_even_if_counter_matches_old_value(self):
        for previous in ("0", "1", "7"):
            with self.subTest(previous=previous):
                result = self.run_function("healthy", restart_first=True,
                                           UPDATE_TEST_RESTARTS_BEFORE=previous,
                                           UPDATE_TEST_RESTARTS_AFTER="1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("alisa-telegram не держится", result.stderr)

    def test_inactive_service_is_rejected_without_any_restarts(self):
        result = self.run_function("healthy", restart_first=True, UPDATE_TEST_SERVICE_STATE="failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("alisa-telegram не держится", result.stderr)


if __name__ == "__main__":
    unittest.main()
