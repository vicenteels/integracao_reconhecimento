"""Executa face-api em Chrome sem janela; a câmera e a UI ficam no OpenCV."""
import json
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
WEB_PROJECT = ROOT.parent / "facial-recognition"
FACE_API_JS = WEB_PROJECT / "node_modules/@vladmandic/face-api/dist/face-api.js"
MODELS = WEB_PROJECT / "public/models/face-api"
MODEL_FILES = {
    "ssd_mobilenetv1_model-weights_manifest.json", "ssd_mobilenetv1_model.bin",
    "face_landmark_68_model-weights_manifest.json", "face_landmark_68_model.bin",
    "face_recognition_model-weights_manifest.json", "face_recognition_model.bin",
}
CHROME_PATHS = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)


class FaceApiWorker:
    def __init__(self):
        if not FACE_API_JS.is_file() or not all((MODELS / file).is_file() for file in MODEL_FILES):
            raise RuntimeError("Modelos face-api não encontrados no projeto web vizinho.")
        self.browser_path = next((path for path in CHROME_PATHS if path.is_file()), None)
        if self.browser_path is None:
            raise RuntimeError("Chrome ou Edge não encontrado para calcular os embeddings face-api.")
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.result = None
        self.error = None
        self.ready = False
        self.server = None
        self.server_thread = None
        self.browser = None
        self.profile = None

    def start(self):
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send_bytes(self, body, content_type, status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    self.send_bytes((ROOT / "face_api_worker.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/face-api.js":
                    self.send_bytes(FACE_API_JS.read_bytes(), "text/javascript; charset=utf-8")
                elif path.startswith("/models/") and path[8:] in MODEL_FILES:
                    file = MODELS / path[8:]
                    self.send_bytes(file.read_bytes(), "application/json" if file.suffix == ".json" else "application/octet-stream")
                elif path == "/frame":
                    with worker.lock:
                        frame, sequence = worker.frame, worker.sequence
                    if frame is None:
                        self.send_bytes(b"", "application/octet-stream", 204)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("X-Frame-Sequence", str(sequence))
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path not in ("/result", "/status"):
                    self.send_error(404)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size > 100000 or size < 2:
                        raise ValueError()
                    payload = json.loads(self.rfile.read(size))
                    with worker.lock:
                        if self.path == "/status":
                            worker.ready = bool(payload.get("ready"))
                            worker.error = str(payload.get("error"))[:200] if payload.get("error") else None
                        else:
                            worker.result = payload
                    self.send_bytes(b"{}", "application/json")
                except (ValueError, TypeError, json.JSONDecodeError):
                    self.send_error(400)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.profile = tempfile.TemporaryDirectory(prefix="face-api-worker-")
        url = f"http://127.0.0.1:{self.server.server_port}/"
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        try:
            self.browser = subprocess.Popen([
                str(self.browser_path), "--headless=new", "--no-first-run", "--no-default-browser-check",
                "--disable-extensions", "--disable-background-networking",
                f"--user-data-dir={self.profile.name}", url,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        except OSError:
            self.close()
            raise RuntimeError("Não foi possível iniciar o processamento face-api local.") from None

    def submit(self, jpeg):
        with self.lock:
            self.sequence += 1
            self.frame = jpeg

    def snapshot(self):
        with self.lock:
            return self.ready, self.error, self.result

    def close(self):
        if self.browser is not None:
            self.browser.terminate()
            try:
                self.browser.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.browser.kill()
            self.browser = None
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.profile is not None:
            self.profile.cleanup()
            self.profile = None
