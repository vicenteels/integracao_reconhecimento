"""Janela OpenCV local com descritores face-api do cadastro web."""
import math
import threading
import time

import cv2

from data import fetch_web_references
from face_api_worker import FaceApiWorker


def match_teacher(descriptor, references, threshold=0.6):
    if len(descriptor) != 128 or any(not math.isfinite(value) for value in descriptor):
        return None
    best = None
    for teacher in references:
        for reference in teacher["embedding_fotos"]:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(descriptor, reference)))
            if best is None or distance < best[1]:
                best = teacher["nome"], distance
    return best if best and best[1] < threshold else None


class WebReferenceCache:
    def __init__(self, database_url, interval=30):
        self.database_url = database_url
        self.interval = interval
        self.references = []
        self.status = "Aguardando sincronização"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def refresh(self):
        try:
            refs, invalid = fetch_web_references(self.database_url)
            with self.lock:
                self.references = refs
                self.status = f"Sincronizado: {len(refs)} professor(es); {invalid} referência(s) inválida(s)"
        except RuntimeError:
            with self.lock:
                self.status = "Integração indisponível; usando referências anteriores" if self.references else "Integração indisponível; sem referências"

    def start(self):
        self.refresh()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.refresh()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)


def recognize(database_url, camera_index=0):
    cache = WebReferenceCache(database_url)
    cache.start()
    worker = FaceApiWorker()
    camera = None
    try:
        worker.start()
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"Câmera {camera_index} indisponível. Verifique índice e permissões.")
        print("Reconhecimento local iniciado. Pressione q para sair.")
        last_submit = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Falha ao ler a câmera.")
            now = time.monotonic()
            if now - last_submit >= 0.5:
                encoded, jpeg = cv2.imencode(".jpg", frame)
                if encoded:
                    worker.submit(jpeg.tobytes())
                last_submit = now
            ready, error, result = worker.snapshot()
            with cache.lock:
                references = cache.references
                status = cache.status
            if error:
                raise RuntimeError(f"Falha no processamento face-api: {error}")
            label_status = "Processando face-api" if ready else "Carregando modelos face-api"
            cv2.putText(frame, label_status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)
            cv2.putText(frame, status[:90], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
            if result and references:
                for face in result.get("faces", []):
                    box = face.get("box", {})
                    x, y, w, h = (int(box.get(key, 0)) for key in ("x", "y", "width", "height"))
                    found = match_teacher(face.get("descriptor", []), references)
                    label = found[0] if found else "Desconhecido"
                    color = (0, 180, 0) if found else (0, 0, 220)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, max(y - 8, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            elif not references:
                cv2.putText(frame, "Sem referencias faciais", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            cv2.imshow("Reconhecimento facial", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cache.close()
        worker.close()
        if camera:
            camera.release()
        cv2.destroyAllWindows()
