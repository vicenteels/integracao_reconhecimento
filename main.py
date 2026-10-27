"""Cadastro biométrico explícito e reconhecimento local pela câmera."""
import argparse
import os
import threading
from pathlib import Path

import cv2
from dotenv import load_dotenv

from core import identify, teacher_id
from data import PostgresStore, fetch_teachers
from vision import Vision
from local_recognition import recognize as recognize_local

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env.local")


def settings():
    model_dir = Path(os.environ.get("MODEL_DIR", "modelo"))
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    return model_dir, os.environ.get("DATABASE_URL")


def open_camera(index):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Câmera {index} indisponível. Verifique índice e permissões.")
    return camera


def enroll(args):
    teachers = fetch_teachers(os.environ.get("DATABASE_URL"))
    selected = teacher_id(args.uuid)
    teacher = teachers.get(selected)
    if not teacher:
        raise RuntimeError("UUID não encontrado entre os professores cadastrados no sistema web.")
    print(f"Professor: {teacher.nome} | {teacher.email} | UUID: {teacher.id}")
    if input("Confirma o cadastro biométrico deste professor? [s/N] ").strip().lower() != "s":
        print("Cadastro cancelado.")
        return
    model_dir, database_url = settings()
    store = PostgresStore(database_url)
    store.check_ready()
    vision = Vision(model_dir)
    camera = open_camera(args.camera)
    vectors = []
    print("Pressione ESPAÇO para cada uma das cinco capturas; q para cancelar. Mude ligeiramente o ângulo.")
    try:
        while len(vectors) < 5:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Falha ao ler a câmera.")
            faces = vision.faces(frame)
            cv2.putText(frame, f"Capturas: {len(vectors)}/5 | rostos: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            cv2.imshow("Cadastro biometrico", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Cadastro cancelado.")
                return
            if key == 32:
                if len(faces) != 1:
                    print("Captura recusada: é necessário exatamente um rosto.")
                    continue
                vectors.append(vision.embedding(frame, faces[0]))
                print(f"Captura válida {len(vectors)}/5")
        # Reconfere o vínculo antes da gravação; professor excluído não recebe referências.
        if selected not in fetch_teachers(os.environ.get("DATABASE_URL")):
            raise RuntimeError("Professor removido durante a captura. Nada foi salvo.")
        store.save(selected, vectors)
        print("Cinco referências SFace gravadas no PostgreSQL atual.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


class ReferenceCache:
    def __init__(self, store, url, interval):
        self.store, self.url, self.interval = store, url, interval
        self.references = []
        self.status = "Aguardando sincronização"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def refresh(self):
        try:
            teachers = fetch_teachers(self.url)
            refs, invalid = self.store.load(teachers)
            with self.lock:
                self.references = refs
                self.status = f"Sincronizado: {len(refs)} professor(es); {invalid} referência(s) inválida(s)"
        except Exception:
            with self.lock:
                self.status = (
                    "Integração indisponível; usando referências carregadas anteriormente"
                    if self.references else "Integração indisponível; nenhuma referência carregada"
                )

    def start(self):
        self.refresh()

        def loop():
            while not self.stop.wait(self.interval):
                self.refresh()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)


def recognize_sface(args):
    model_dir, database_url = settings()
    vision = Vision(model_dir)
    cache = ReferenceCache(PostgresStore(database_url), database_url, args.sync_interval)
    camera = None
    try:
        cache.start()
        camera = open_camera(args.camera)
        print("Reconhecimento iniciado. Pressione q para sair.")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Falha ao ler a câmera.")
            faces = vision.faces(frame)
            with cache.lock:
                references = cache.references
                status = cache.status
            for face in faces:
                x, y, w, h = map(int, face[:4])
                found = identify(vision.embedding(frame, face), references, args.threshold, args.strategy) if references else None
                label = found[0].nome if found else "Desconhecido"
                color = (0, 180, 0) if found else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if not references:
                cv2.putText(frame, "Sem referencias faciais", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
            cv2.putText(frame, status[:90], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
            cv2.imshow("Reconhecimento facial", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cache.close()
        if camera:
            camera.release()
        cv2.destroyAllWindows()


def recognize(args):
    _, database_url = settings()
    if not database_url:
        raise RuntimeError("DATABASE_URL não configurada.")
    recognize_local(database_url, args.camera)


def main():
    parser = argparse.ArgumentParser(description="Integração de professores e reconhecimento SFace")
    sub = parser.add_subparsers(dest="mode", required=True)
    enrollment = sub.add_parser("cadastrar", help="capturar cinco rostos para um UUID confirmado")
    enrollment.add_argument("uuid")
    recognition = sub.add_parser("reconhecer", help="abrir câmera local e reconhecer com face-api")
    recognition.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")))
    enrollment.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")))
    args = parser.parse_args()
    try:
        (enroll if args.mode == "cadastrar" else recognize)(args)
    except (RuntimeError, ValueError, OSError) as exc:
        # Não imprime exceções do driver, que poderiam conter credenciais.
        print(f"Erro: {exc}" if not isinstance(exc, OSError) else "Erro de integração ou arquivos; confira a configuração local.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
