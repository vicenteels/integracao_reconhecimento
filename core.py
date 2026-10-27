"""Vetores SFace; nunca aceita descritores face-api do cadastro web."""
import math
import uuid
from dataclasses import dataclass

DIMENSION = 128
COUNT = 5
MODEL_ID = "opencv_sface_2021dec_yunet_2023mar_normalized"


@dataclass(frozen=True)
class Teacher:
    id: str
    nome: str
    email: str


def teacher_id(value):
    return str(uuid.UUID(str(value)))


def validate_vectors(vectors):
    if not isinstance(vectors, (list, tuple)) or len(vectors) != COUNT:
        raise ValueError("São necessários exatamente cinco embeddings.")
    clean = []
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or len(vector) != DIMENSION:
            raise ValueError("Cada embedding precisa ter 128 valores.")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
            raise ValueError("Embedding contém valor não numérico ou não finito.")
        norm = math.sqrt(sum(float(x) ** 2 for x in vector))
        if not 0.98 <= norm <= 1.02:
            raise ValueError("Embedding SFace deve estar normalizado.")
        clean.append(tuple(float(x) for x in vector))
    return tuple(clean)


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.dist(a, [0.0] * DIMENSION) * math.dist(b, [0.0] * DIMENSION))


def identify(query, references, threshold=0.45, strategy="max"):
    if len(query) != DIMENSION or not all(math.isfinite(float(x)) for x in query):
        raise ValueError("Embedding de consulta inválido.")
    if strategy not in ("max", "mean_distance", "mean_vector"):
        raise ValueError("Estratégia inválida.")
    best = None
    for teacher, vectors in references:
        vectors = validate_vectors(vectors)
        scores = [cosine(query, v) for v in vectors]
        if strategy == "max":
            score = max(scores)
        elif strategy == "mean_distance":
            score = sum(scores) / COUNT
        else:
            mean = [sum(v[i] for v in vectors) / COUNT for i in range(DIMENSION)]
            score = cosine(query, mean)
        if best is None or score > best[1]:
            best = (teacher, score)
    return best if best and best[1] >= threshold else None
