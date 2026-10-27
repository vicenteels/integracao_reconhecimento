"""Acesso somente leitura aos professores e referências legadas SFace."""

from core import MODEL_ID, Teacher, teacher_id, validate_vectors
import math


def fetch_teachers(database_url):
    if not database_url:
        raise RuntimeError("DATABASE_URL não configurada.")
    import psycopg
    # A credencial deve ter apenas SELECT nas três colunas; transação read-only.
    try:
        with psycopg.connect(database_url, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, nome, email FROM professores ORDER BY nome, id")
                rows = cur.fetchall()
    except psycopg.Error:
        raise RuntimeError("Não foi possível consultar professores no PostgreSQL central.") from None
    return {teacher_id(row[0]): Teacher(teacher_id(row[0]), row[1], row[2]) for row in rows}


def fetch_web_references(database_url):
    """Retorna apenas descritores face-api íntegros da tabela existente."""
    if not database_url:
        raise RuntimeError("DATABASE_URL não configurada.")
    import psycopg
    try:
        with psycopg.connect(database_url, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, nome, embedding_fotos FROM professores ORDER BY nome, id")
                rows = cur.fetchall()
    except psycopg.Error:
        raise RuntimeError("Não foi possível carregar os embeddings de professores.") from None
    references = []
    invalid = 0
    for uuid_value, name, vectors in rows:
        if (not isinstance(vectors, (list, tuple)) or len(vectors) != 5
                or any(not isinstance(v, (list, tuple)) or len(v) != 128
                       or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v)
                       for v in vectors)):
            invalid += 1
            continue
        references.append({"id": teacher_id(uuid_value), "nome": name,
                           "embedding_fotos": vectors})
    return references, invalid


def parse_reference_rows(rows, teachers):
    references = []
    invalid = 0
    for uuid_value, model, vectors in rows:
        try:
            uuid_ = teacher_id(uuid_value)
        except ValueError:
            invalid += 1
            continue
        if uuid_ not in teachers:
            continue
        try:
            if model != MODEL_ID:
                raise ValueError("Modelo incompatível")
            references.append((teachers[uuid_], validate_vectors(vectors)))
        except (ValueError, TypeError):
            invalid += 1
    return references, invalid


class PostgresStore:
    def __init__(self, database_url):
        if not database_url:
            raise RuntimeError("DATABASE_URL não configurada.")
        self.database_url = database_url

    def check_ready(self):
        import psycopg
        try:
            with psycopg.connect(self.database_url, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM professor_sface_referencias LIMIT 0")
        except psycopg.Error:
            raise RuntimeError("Tabela professor_sface_referencias indisponível; veja criar_tabela_sface.sql.") from None

    def load(self, teachers):
        import psycopg
        try:
            with psycopg.connect(self.database_url, connect_timeout=5, options="-c default_transaction_read_only=on") as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT professor_id, model_id, embedding_fotos FROM professor_sface_referencias")
                    rows = cur.fetchall()
        except psycopg.Error:
            raise RuntimeError("Não foi possível carregar referências SFace; confira a tabela professor_sface_referencias.") from None
        return parse_reference_rows(rows, teachers)

    def save(self, teacher_uuid, vectors):
        import psycopg
        uuid_ = teacher_id(teacher_uuid)
        clean = validate_vectors(vectors)
        try:
            # A transação grava as cinco referências ou nenhuma.
            with psycopg.connect(self.database_url, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO professor_sface_referencias (professor_id, model_id, embedding_fotos) "
                        "VALUES (%s, %s, %s) ON CONFLICT (professor_id) DO UPDATE SET "
                        "model_id = EXCLUDED.model_id, embedding_fotos = EXCLUDED.embedding_fotos, "
                        "atualizado_em = CURRENT_TIMESTAMP",
                        (uuid_, MODEL_ID, [list(vector) for vector in clean]),
                    )
        except psycopg.Error:
            raise RuntimeError("Não foi possível gravar referências SFace; confira a tabela e as permissões.") from None
