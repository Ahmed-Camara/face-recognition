import cv2 as cv
import os
import uuid

def save_face(output_face_dir,photo, prefix: str) -> str:
    """Enregistre le visage extrait et retourne son chemin."""
    path = os.path.join(output_face_dir, f"{prefix}_{uuid.uuid4().hex}.jpg")
    if not cv.imwrite(path, photo, [cv.IMWRITE_JPEG_QUALITY, 95]):
        raise IOError(f"Impossible d'enregistrer l'image: {path}")
    return path