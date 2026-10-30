"""Vérification selfie / pièce d'identité, entièrement en mémoire."""
import numpy as np

from src.face_engine import FaceInfo, get_engine
from utils.config import load_config
from utils.image_io import decode_image


def compare_faces(id_face: FaceInfo, selfie_face: FaceInfo, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    match_t = float(cfg["decision"]["match_threshold"])
    review_t = float(cfg["decision"]["review_threshold"])

    # Embeddings normalisés : le produit scalaire est la similarité cosinus
    similarity = float(np.dot(id_face.embedding, selfie_face.embedding))

    if similarity >= match_t:
        decision = "match"
    elif similarity >= review_t:
        decision = "review"          # zone grise : contrôle humain recommandé
    else:
        decision = "no_match"

    return {
        "decision": decision,
        "match": decision == "match",
        "decision":decision,
        "similarity": round(similarity, 4),
        "thresholds": {"match": match_t, "review": review_t},
        "model": cfg["model"]["pack"],
        "id_card": id_face.public(),
        "selfie": selfie_face.public(),
        "warnings": id_face.warnings + selfie_face.warnings,
    }


def verify_selfie_against_id(id_card: bytes, selfie: bytes) -> dict:
    """Pipeline complet à partir des octets reçus.

    Raises:
        FaceVerificationError (et sous-classes) : erreur métier à montrer au client.
    """
    cfg = load_config()
    max_side = int(cfg["img"]["max_side"])
    engine = get_engine()

    id_img = decode_image(id_card, max_side, field="id_card")
    selfie_img = decode_image(selfie, max_side, field="selfie")

    id_face = engine.extract(id_img, "id_card", "la pièce d'identité")
    selfie_face = engine.extract(selfie_img, "selfie", "le selfie", require_live=True)
    return compare_faces(id_face, selfie_face, cfg)