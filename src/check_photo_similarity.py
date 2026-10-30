import os
import tempfile
 
import cv2
from deepface import DeepFace
 
from exception.exceptions import FaceVerificationError, NoFaceFoundError
from src.detect_deepface import extract_id_photo
from utils.load_config import load_config
 
# --- Réglages (lus dans config/settings.yaml, section model_params) --------
_params = load_config()["model_params"]
MODEL_NAME = _params["model_name"]               # ex: Facenet512
DISTANCE_METRIC = _params["distance_metric"]     # cosine | euclidean | euclidean_l2
VERIFY_DETECTOR = _params["verify_detector"]     # détecteur DeepFace pour la comparaison
EXTRACT_MARGIN = float(_params["margin"])        # marge autour du visage (0.3 = tête + épaules)
# Facteur de sévérité appliqué au seuil du modèle (1.0 = seuil DeepFace, 0.9 = 10 % plus strict).
# Les seuils par défaut de DeepFace visent un équilibre général, pas la vérification d'identité.
THRESHOLD_FACTOR = float(_params.get("threshold_factor", 1.0))
# Détecteur pour extraire le visage de l'image (clé optionnelle, secours automatique sur opencv)
EXTRACT_DETECTOR = _params.get("extract_detector", "retinaface")
 
 
# --- Niveau 1 : comparaison de deux visages déjà recadrés ------------------
def check_photo_similarity(img1_path: str, img2_path: str) -> dict:
    """
    Compare deux images de visage et indique si elles correspondent
    probablement à la même personne.
 
    Args:
        img1_path: chemin de la première image (ex: selfie)
        img2_path: chemin de la seconde image (ex: photo de la pièce d'identité)
 
    Returns:
        {
          "match": bool,              # verdict (distance <= seuil du modèle)
          "distance": float,          # distance brute entre les deux visages
          "threshold": float,         # seuil du modèle pour cette métrique
          "similarity_score": float,  # 0..1 (1 - distance cosinus), pour affichage
          "model": str, "detector": str, "distance_metric": str
        }
 
    Raises:
        FileNotFoundError: un des fichiers n'existe pas
        FaceVerificationError: visage introuvable ou image inexploitable
    """
    for label, path in (("img1_path", img1_path), ("img2_path", img2_path)):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"{label} introuvable: {path}")
 
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=MODEL_NAME,
            detector_backend=VERIFY_DETECTOR,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True,  # on refuse de comparer si aucun visage n'est trouvé
            align=True,
        )
    except ValueError as exc:
        # DeepFace enveloppe l'erreur réelle: "Exception while processing img1_path"
        which = "la première image" if "img1_path" in str(exc) else (
            "la seconde image" if "img2_path" in str(exc) else "une des images"
        )
        cause = exc.__cause__ or exc
        raise FaceVerificationError(
            f"Impossible de détecter un visage exploitable dans {which}: {cause}"
        ) from exc
 
    distance = float(result["distance"])
    threshold = float(result["threshold"]) * THRESHOLD_FACTOR
 
    return {
        "match": distance <= threshold,   # seuil durci (voir THRESHOLD_FACTOR)
        "distance": round(distance, 4),
        "threshold": round(threshold, 4),
        "similarity_score": round(max(0.0, 1.0 - distance / threshold) if threshold else 0.0, 4),
        "model": MODEL_NAME,
        "detector": VERIFY_DETECTOR,
        "distance_metric": DISTANCE_METRIC,
    }
 
 
# --- Niveau 2 : pipeline complet selfie + pièce d'identité -----------------
def _extract_and_save(src_path: str, out_path: str, label: str) -> str:
    """Extrait le visage de src_path et l'enregistre dans out_path."""
    try:
        face = extract_id_photo(
            str(src_path), margin=EXTRACT_MARGIN, detector=EXTRACT_DETECTOR
        )
    except NoFaceFoundError as exc:
        raise NoFaceFoundError(f"Aucun visage détecté sur {label}.") from exc
 
    if not cv2.imwrite(out_path, face, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise FaceVerificationError(f"Impossible d'enregistrer le visage de {label}: {out_path}")
    return out_path
 
 
def verify_selfie_against_id(selfie_path: str, id_card_path: str) -> dict:
    """
    Pipeline complet :
      1. extrait le visage du selfie
      2. extrait la photo de la pièce d'identité
      3. compare les deux visages
      4. supprime les fichiers temporaires (même en cas d'erreur)
 
    Raises:
        FileNotFoundError, NoFaceFoundError, FaceVerificationError
    """
    for label, path in (("selfie", selfie_path), ("pièce d'identité", id_card_path)):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Fichier {label} introuvable: {path}")
 
    # Le dossier temporaire (et les visages dedans) est supprimé à la sortie du bloc.
    with tempfile.TemporaryDirectory(prefix="faces_") as tmp_dir:
        selfie_face = _extract_and_save(
            selfie_path, os.path.join(tmp_dir, "selfie_face.jpg"), "le selfie"
        )
        id_face = _extract_and_save(
            id_card_path, os.path.join(tmp_dir, "id_face.jpg"), "la pièce d'identité"
        )
        return check_photo_similarity(selfie_face, id_face)
"""
from deepface import DeepFace
import cv2 as cv
import os
from exception.exceptions import NoFaceFoundError,FaceVerificationError
from src.detect_deepface import extract_id_photo
from utils.load_config import load_config

config = load_config()

model_name = config["model_params"]["model_name"] #"Facenet512"   "ArcFace",#"VGG-Face",Facenet512
distance_metric = config["model_params"]["distance_metric"]
verify_detector = config["model_params"]["verify_detector"]

def check_photo_similarity(img1_path:str, img2_path:str) -> bool:
    
    for label, path in (("img1_path", img1_path), ("img2_path", img2_path)):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"{label} introuvable: {path}")
    try:
        print(f"filename {img1_path}")
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=verify_detector,
            enforce_detection=True,
            distance_metric=distance_metric,
            align=True
        )
        
        #if result["verified"]:
            #return True
    except ValueError as exc:
        which = "la première image" if "img1_path" in str(exc) else (
            "la seconde image" if "img2_path" in str(exc) else "une des images"
        )
        cause = exc.__cause__ or exc
        raise FaceVerificationError(
            f"Impossible de détecter un visage exploitable dans {which}: {cause}"
        ) from exc
    
    distance = float(result["distance"])
    threshold = float(result["threshold"])
    
    return {
        "match": bool(result["verified"]),
        "distance": round(distance, 4),
        "threshold": round(threshold, 4),
        "similarity_score": round(max(0.0, 1.0 - distance), 4),
        "model": model_name,
        "detector": verify_detector,
        "distance_metric": distance_metric,
    }
    
"""