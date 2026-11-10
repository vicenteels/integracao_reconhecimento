"""Janela OpenCV local com descritores face-api do cadastro web e liberação da sala."""
import math
import threading
import time

import cv2

from data import fetch_web_references, registrar_acesso
from face_api_worker import FaceApiWorker

VERDE = (0, 180, 0)
VERMELHO = (0, 0, 220)
NEGADO = "Acesso facial negado"
# cv2.putText não desenha acentos.
MOTIVOS = {"SEM_RESERVA": "sem reserva neste horario", "OUTRO_PROFESSOR": "reserva de outro professor"}


def match_teacher(descriptor, references, threshold=0.6):
    """Retorna (professor, distância) do professor mais próximo abaixo do limiar."""
    if len(descriptor) != 128 or any(not math.isfinite(value) for value in descriptor):
        return None
    best = None
    for teacher in references:
        for reference in teacher["embedding_fotos"]:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(descriptor, reference)))
            if best is None or distance < best[1]:
                best = teacher, distance
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


class AccessController:
    """Aplica as regras de reserva sem bloquear a janela da câmera.

    Depois de uma tentativa, o mesmo professor fica em espera por `cooldown`
    segundos, evitando consultar o banco a cada quadro.
    """

    def __init__(self, database_url, sala_id, cooldown=15.0, message_seconds=5.0,
                 registrar=registrar_acesso, clock=time.monotonic):
        self.database_url = database_url
        self.sala_id = sala_id
        self.cooldown = cooldown
        self.message_seconds = message_seconds
        self.registrar = registrar
        self.clock = clock
        self.lock = threading.Lock()
        self.busy = False
        self.last_attempt = {}
        self.message = None  # (texto, cor, expira_em)

    def _show(self, text, color):
        with self.lock:
            self.message = (text, color, self.clock() + self.message_seconds)

    def current_message(self):
        with self.lock:
            if self.message and self.message[2] > self.clock():
                return self.message[:2]
            return None

    def deny_unknown(self):
        with self.lock:
            busy = self.busy
            showing = self.message and self.message[2] > self.clock()
        if not busy and not showing:
            self._show(NEGADO, VERMELHO)

    def request(self, teacher, background=True):
        now = self.clock()
        with self.lock:
            if self.busy or now - self.last_attempt.get(teacher["id"], -math.inf) < self.cooldown:
                return False
            self.busy = True
            self.last_attempt[teacher["id"]] = now
        if background:
            threading.Thread(target=self._run, args=(teacher,), daemon=True).start()
        else:
            self._run(teacher)
        return True

    def _run(self, teacher):
        try:
            allowed, detail = self.registrar(self.database_url, self.sala_id, teacher["id"])
            if not allowed:
                self._show(f"{NEGADO}: {MOTIVOS.get(detail, detail)}", VERMELHO)
            elif detail == "EM_USO":
                self._show(f"Sala liberada para {teacher['nome']}", VERDE)
            else:
                self._show(f"Sala ja liberada para {teacher['nome']}", VERDE)
        except Exception:
            with self.lock:
                # Permite nova tentativa logo em seguida se o banco falhou.
                self.last_attempt.pop(teacher["id"], None)
            self._show("Falha ao consultar reservas; tente novamente", (0, 180, 255))
        finally:
            with self.lock:
                self.busy = False


def recognize(database_url, sala_id, camera_index=0):
    cache = WebReferenceCache(database_url)
    cache.start()
    access = AccessController(database_url, sala_id)
    worker = FaceApiWorker()
    camera = None
    try:
        worker.start()
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError(f"Câmera {camera_index} indisponível. Verifique índice e permissões.")
        print("Reconhecimento local iniciado. Pressione q para sair.")
        last_submit = 0
        last_sequence = None
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
                # Decide o acesso só uma vez por resultado novo do face-api.
                new_result = result.get("sequence") != last_sequence
                last_sequence = result.get("sequence")
                known, unknown = [], False
                for face in result.get("faces", []):
                    box = face.get("box", {})
                    x, y, w, h = (int(box.get(key, 0)) for key in ("x", "y", "width", "height"))
                    found = match_teacher(face.get("descriptor", []), references)
                    if found:
                        known.append(found)
                    else:
                        unknown = True
                    label = found[0]["nome"] if found else "Desconhecido"
                    color = VERDE if found else VERMELHO
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, max(y - 8, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                if new_result:
                    if known:
                        access.request(min(known, key=lambda item: item[1])[0])
                    elif unknown:
                        access.deny_unknown()
            elif not references:
                cv2.putText(frame, "Sem referencias faciais", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            message = access.current_message()
            if message:
                text, color = message
                height = frame.shape[0]
                cv2.rectangle(frame, (0, height - 40), (frame.shape[1], height), color, -1)
                cv2.putText(frame, text[:70], (10, height - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Reconhecimento facial", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cache.close()
        worker.close()
        if camera:
            camera.release()
        cv2.destroyAllWindows()
