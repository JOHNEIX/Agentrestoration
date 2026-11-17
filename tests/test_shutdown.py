import signal

import pytest

from ui import service


def test_shutdown_uses_server_handler_and_is_idempotent(monkeypatch):
    callbacks = []
    scheduled = []
    monkeypatch.setattr(service, "_shutdown_pending", False)
    monkeypatch.setattr(service.signal, "getsignal", lambda kind: lambda *args: callbacks.append(args))
    class Timer:
        def __init__(self, delay, handler, args):
            self.handler, self.args = handler, args
            assert delay > 0
        def start(self): scheduled.append(self)
    monkeypatch.setattr(service.threading, "Timer", Timer)
    service.shutdown()
    service.shutdown()
    assert len(scheduled) == 1
    assert scheduled[0].daemon
    assert not callbacks
    scheduled[0].handler(*scheduled[0].args)
    assert callbacks == [(signal.SIGINT, None)]


def test_shutdown_refuses_active_task_and_releases_counter(monkeypatch):
    monkeypatch.setattr(service, "_shutdown_pending", False)
    def processing(*args):
        with pytest.raises(RuntimeError, match="正在修复"):
            service.shutdown()
        raise ValueError("simulated task failure")
    monkeypatch.setattr(service, "_execute", processing)
    with pytest.raises(ValueError):
        service.execute(None, None, None, None)
    assert service._active_jobs == 0
    assert not service._shutdown_pending


def test_shutdown_requires_server_handler(monkeypatch):
    monkeypatch.setattr(service, "_shutdown_pending", False)
    monkeypatch.setattr(service.signal, "getsignal", lambda kind: signal.default_int_handler)
    with pytest.raises(RuntimeError, match="关闭接口"):
        service.shutdown()
    assert not service._shutdown_pending


def test_real_server_shutdown_releases_port(tmp_path):
    import os
    import socket
    import subprocess
    import sys
    from agentrestoration.core import ROOT
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    script = tmp_path / "shutdown_server.py"
    script.write_text("""
import threading, time, urllib.request
from streamlit.web import bootstrap
from agentrestoration.core import ROOT
from ui.service import shutdown

def stop_when_ready():
    for attempt in range(200):
        try:
            with urllib.request.urlopen('http://127.0.0.1:PORT/_stcore/health', timeout=.2) as response:
                ready = response.read() == b'ok'
        except Exception:
            ready = False
        if ready:
            shutdown()
            print('SHUTDOWN_REQUESTED', flush=True)
            return
        time.sleep(.05)

options = {
    'server.port': PORT, 'server.address': '127.0.0.1',
    'server.headless': True, 'browser.gatherUsageStats': False}
bootstrap.load_config_options(options)
threading.Thread(target=stop_when_ready, daemon=True).start()
bootstrap.run(str(ROOT / 'ui/app.py'), False, [], options)
""".replace("PORT", str(port)), encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(script)], cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT)),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    try:
        output, _ = process.communicate(timeout=25)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 0, output
    assert "SHUTDOWN_REQUESTED" in output
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1", port)) != 0
