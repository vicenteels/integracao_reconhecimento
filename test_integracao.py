import math
import sys
import types
import unittest
import uuid
from unittest.mock import MagicMock, patch

from core import Teacher, identify, validate_vectors
from core import MODEL_ID
from data import PostgresStore, parse_reference_rows, fetch_web_references
from main import ReferenceCache, open_camera
from data import registrar_acesso
from local_recognition import AccessController, match_teacher


def unit(index):
    v = [0.0] * 128
    v[index] = 1.0
    return v


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.id = str(uuid.uuid4())
        self.teacher = Teacher(self.id, "Professor Teste", "teste@example.org")
        self.teachers = {self.id: self.teacher}

    def test_uuid_link_and_match(self):
        rows = [(self.id, MODEL_ID, [unit(0)] * 5)]
        refs, invalid = parse_reference_rows(rows, self.teachers)
        self.assertEqual(invalid, 0)
        self.assertEqual(identify(unit(0), refs)[0].id, self.id)
        self.assertIsNone(identify(unit(1), refs))
        self.assertEqual(parse_reference_rows(rows, {})[0], [])

    def test_reject_bad_vectors_and_model(self):
        for vectors in ([unit(0)], [unit(0)] * 4 + [[math.nan] * 128], [unit(0)] * 4 + [[1.0] * 127]):
            with self.assertRaises(ValueError):
                validate_vectors(vectors)
        rows = [(self.id, MODEL_ID, [unit(0)] * 4)]
        refs, invalid = parse_reference_rows(rows, self.teachers)
        self.assertEqual((refs, invalid), ([], 1))
        refs, invalid = parse_reference_rows([(self.id, "face-api", [unit(0)] * 5)], self.teachers)
        self.assertEqual((refs, invalid), ([], 1))

    def test_refresh_rename_delete_failure(self):
        rows = [(self.id, MODEL_ID, [unit(0)] * 5)]
        class Store:
            def load(_, teachers):
                return parse_reference_rows(rows, teachers)
        cache = ReferenceCache(Store(), "unused", 30)
        with patch("main.fetch_teachers", return_value=self.teachers):
            cache.refresh()
        self.assertEqual(cache.references[0][0].nome, "Professor Teste")
        renamed = Teacher(self.id, "Nome Novo", "novo@example.org")
        with patch("main.fetch_teachers", return_value={self.id: renamed}):
            cache.refresh()
        self.assertEqual(cache.references[0][0].nome, "Nome Novo")
        with patch("main.fetch_teachers", side_effect=RuntimeError("offline")):
            cache.refresh()
        self.assertEqual(cache.references[0][0].nome, "Nome Novo")
        self.assertIn("anteriormente", cache.status)
        with patch("main.fetch_teachers", return_value={}):
            cache.refresh()
        self.assertEqual(cache.references, [])

    def test_postgres_save_parameterized_five_vectors(self):
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        fake = types.SimpleNamespace(connect=MagicMock(return_value=connection), Error=Exception)
        with patch.dict(sys.modules, {"psycopg": fake}):
            PostgresStore("postgresql://example").save(self.id, [unit(0)] * 5)
        self.assertEqual(cursor.execute.call_count, 1)
        sql, params = cursor.execute.call_args.args
        self.assertIn("VALUES (%s, %s, %s)", sql)
        self.assertEqual(params[0], self.id)
        self.assertEqual(params[1], MODEL_ID)
        self.assertEqual(len(params[2]), 5)
        self.assertTrue(all(len(vector) == 128 for vector in params[2]))

    def test_camera_unavailable(self):
        class Camera:
            released = False
            def isOpened(self): return False
            def release(self): self.released = True
        camera = Camera()
        with patch("main.cv2.VideoCapture", return_value=camera):
            with self.assertRaisesRegex(RuntimeError, "indisponível"):
                open_camera(99)
        self.assertTrue(camera.released)

    def test_web_references_from_existing_professores_table(self):
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            (self.id, "Professor Teste", [unit(0)] * 5),
            (str(uuid.uuid4()), "Invalido", [unit(0)] * 4),
        ]
        fake = types.SimpleNamespace(connect=MagicMock(return_value=connection), Error=Exception)
        with patch.dict(sys.modules, {"psycopg": fake}):
            refs, invalid = fetch_web_references("postgresql://example")
        self.assertEqual((len(refs), invalid), (1, 1))
        self.assertEqual(refs[0]["id"], self.id)
        self.assertIn("embedding_fotos FROM professores", cursor.execute.call_args.args[0])

    def test_local_face_api_distance_threshold(self):
        refs = [{"nome": "Professor Teste", "embedding_fotos": [unit(0)] * 5}]
        self.assertEqual(match_teacher(unit(0), refs)[0]["nome"], "Professor Teste")
        self.assertIsNone(match_teacher(unit(1), refs))


    def _registrar(self, rows, professor):
        connection = MagicMock()
        cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = rows
        fake = types.SimpleNamespace(connect=MagicMock(return_value=connection), Error=Exception)
        with patch.dict(sys.modules, {"psycopg": fake}):
            result = registrar_acesso("postgresql://example", str(uuid.uuid4()), professor)
        return result, cursor

    def test_reserva_sem_reserva_e_outro_professor_negam(self):
        (result, cursor) = self._registrar([], self.id)
        self.assertEqual(result, (False, "SEM_RESERVA"))
        self.assertEqual(cursor.execute.call_count, 1)
        self.assertIn("FOR UPDATE", cursor.execute.call_args.args[0])
        outro = str(uuid.uuid4())
        (result, cursor) = self._registrar([("r1", outro, "RESERVA_ATIVA")], self.id)
        self.assertEqual(result, (False, "OUTRO_PROFESSOR"))
        self.assertEqual(cursor.execute.call_count, 1)

    def test_reserva_ativa_vira_em_uso(self):
        rows = [("outra", str(uuid.uuid4()), "RESERVA_ATIVA"), ("minha", self.id, "RESERVA_ATIVA")]
        (result, cursor) = self._registrar(rows, self.id)
        self.assertEqual(result, (True, "EM_USO"))
        sql, params = cursor.execute.call_args.args
        self.assertIn("SET status_reserva = 'EM_USO'", sql)
        self.assertEqual(params, ("minha",))

    def test_reserva_em_uso_nao_e_finalizada(self):
        (result, cursor) = self._registrar([("minha", self.id, "EM_USO")], self.id)
        self.assertEqual(result, (True, "JA_EM_USO"))
        self.assertEqual(cursor.execute.call_count, 1)
        self.assertNotIn("FINALIZADO", cursor.execute.call_args.args[0])

    def test_controle_de_acesso_espera_entre_reconhecimentos(self):
        now = [100.0]
        calls = []
        def registrar(_url, _sala, professor):
            calls.append(professor)
            return True, "EM_USO" if len(calls) == 1 else "JA_EM_USO"
        access = AccessController("url", "sala", cooldown=15, registrar=registrar, clock=lambda: now[0])
        teacher = {"id": self.id, "nome": "Professor Teste"}
        self.assertTrue(access.request(teacher, background=False))
        self.assertIn("Sala liberada", access.current_message()[0])
        now[0] += 1
        self.assertFalse(access.request(teacher, background=False))
        now[0] += 15
        self.assertTrue(access.request(teacher, background=False))
        self.assertIn("ja liberada", access.current_message()[0])
        self.assertEqual(len(calls), 2)

    def test_controle_de_acesso_negado(self):
        access = AccessController("url", "sala", registrar=lambda *_: (False, "OUTRO_PROFESSOR"), clock=lambda: 0.0)
        access.request({"id": self.id, "nome": "X"}, background=False)
        self.assertEqual(access.current_message()[0], "Acesso facial negado: reserva de outro professor")
        unknown = AccessController("url", "sala", clock=lambda: 0.0)
        unknown.deny_unknown()
        self.assertEqual(unknown.current_message()[0], "Acesso facial negado")


if __name__ == "__main__":
    unittest.main()
