from __future__ import annotations

import http.server
import io
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from frog_classifier.app import launch
from frog_classifier.app.paths import MODELS_VARIABLE, ROOT_VARIABLE
from tests.inference.helpers import build_saved_run


APP = Path(__file__).resolve().parents[2] / "classifier_app" / "app.py"


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/_stcore/health" else 404)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass


def serve_health(test: unittest.TestCase) -> int:
    """A stand-in for a running app: it answers the health address only."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)
    return server.server_address[1]


class LaunchPieceTests(unittest.TestCase):
    def test_streamlit_runs_for_this_computer_only_without_developer_tools(self) -> None:
        app_file = Path("C:/scanner/app/classifier_app/app.py")

        command = launch.streamlit_command(app_file, 8571)

        self.assertEqual(command[:6], [sys.executable, "-I", "-m", "streamlit", "run", str(app_file)])
        self.assertEqual(dict(zip(command[6::2], command[7::2])), {
            "--server.port": "8571",
            "--server.address": "localhost",
            "--server.headless": "true",
            "--browser.gatherUsageStats": "false",
            "--client.toolbarMode": "minimal",
            "--server.fileWatcherType": "none",
        })

    def test_python_settings_from_other_software_are_dropped(self) -> None:
        environment = launch.child_environment(
            {"PATH": "C:/Windows", "PYTHONPATH": "C:/gis/python", "PythonHome": "C:/gis", "BROWSER": "edge"},
            Path("C:/data"),
            Path("C:/scanner/app/models"),
        )

        self.assertEqual(environment, {
            "PATH": "C:/Windows",
            "BROWSER": "edge",
            "PYTHONNOUSERSITE": "1",
            ROOT_VARIABLE: str(Path("C:/data")),
            MODELS_VARIABLE: str(Path("C:/scanner/app/models")),
        })

    def test_a_free_preferred_port_is_used(self) -> None:
        port = launch.free_port()

        self.assertEqual(launch.choose_port(port), (port, False))

    def test_a_port_held_by_another_program_is_replaced(self) -> None:
        with socket.socket() as holder:
            holder.bind(("127.0.0.1", 0))
            holder.listen()
            held = holder.getsockname()[1]

            port, running = launch.choose_port(held)

        self.assertNotEqual(port, held)
        self.assertFalse(running)

    def test_an_app_already_answering_is_reused(self) -> None:
        port = serve_health(self)

        self.assertEqual(launch.choose_port(port), (port, True))

    def test_a_configured_internet_proxy_is_not_used_for_the_local_app(self) -> None:
        port = serve_health(self)
        proxies = {"HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9", "NO_PROXY": "", "no_proxy": ""}

        with patch.dict(os.environ, proxies):
            self.assertTrue(launch.app_answers(port))

    def test_a_missing_cpp_runtime_is_explained(self) -> None:
        with patch("ctypes.WinDLL", side_effect=OSError("not found"), create=True):
            problem = launch.runtime_problem()

        self.assertIn("Visual C++ Redistributable (x64)", problem)
        self.assertIn(launch.RUNTIME_ADDRESS, problem)

    def test_this_computer_has_the_cpp_runtime(self) -> None:
        self.assertIsNone(launch.runtime_problem())

    def test_a_slow_start_gives_up_with_a_message(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(process.wait)
        self.addCleanup(process.kill)

        with self.assertRaisesRegex(launch.LaunchError, "did not answer within"):
            launch.wait_until_ready(launch.free_port(), process, timeout_s=0.5)

    def test_a_failed_start_is_reported_at_once(self) -> None:
        process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])

        with self.assertRaisesRegex(launch.LaunchError, "stopped while starting"):
            launch.wait_until_ready(launch.free_port(), process, timeout_s=60)


class LaunchMainTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.data = self.root / "data"

    def _main(self, *arguments: str) -> tuple[int, str]:
        output = io.StringIO()
        with patch("sys.stdout", output):
            code = launch.main([
                "--app", str(APP), "--models", str(self.models), "--data", str(self.data), *arguments,
            ])
        return code, output.getvalue()

    def test_a_running_app_is_opened_instead_of_started_again(self) -> None:
        port = serve_health(self)

        with patch("webbrowser.open", return_value=True) as browser, patch.object(launch, "start_app") as start:
            code, output = self._main("--port", str(port))

        self.assertEqual(code, 0)
        self.assertIn("already running", output)
        browser.assert_called_once_with(f"http://localhost:{port}")
        start.assert_not_called()

    def test_a_missing_runtime_stops_before_anything_starts(self) -> None:
        with patch.object(launch, "runtime_problem", return_value="no runtime here"), \
                patch.object(launch, "start_app") as start:
            code, output = self._main()

        self.assertEqual(code, 1)
        self.assertIn("cannot start: no runtime here.", output)
        start.assert_not_called()

    def test_a_data_folder_that_is_not_a_full_path_is_refused(self) -> None:
        self.data = Path("Frog Call Scanner")

        with patch.object(launch, "start_app") as start:
            code, output = self._main()

        self.assertEqual(code, 1)
        self.assertIn("is not a full path", output)
        start.assert_not_called()

    def test_a_missing_app_file_asks_for_the_whole_package(self) -> None:
        output = io.StringIO()
        with patch("sys.stdout", output), patch.object(launch, "start_app") as start:
            code = launch.main([
                "--app", str(self.root / "missing.py"), "--models", str(self.models), "--data", str(self.data),
            ])

        self.assertEqual(code, 1)
        self.assertIn("unzip the whole package again", output.getvalue())
        start.assert_not_called()


class AppStartTests(unittest.TestCase):
    def test_the_app_starts_for_this_computer_and_answers(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        build_saved_run(root / "models")
        log = (root / "streamlit.log").open("wb")
        self.addCleanup(log.close)
        port = launch.free_port()

        process = launch.start_app(APP, port, launch.child_environment(os.environ, root / "data", root / "models"), log=log)
        self.addCleanup(process.wait, 30)
        self.addCleanup(process.terminate)
        launch.wait_until_ready(port, process)

        self.assertTrue(launch.app_answers(port))
        self.assertFalse(launch.port_is_free(port))
