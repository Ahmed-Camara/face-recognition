"""Décodage des images en mémoire (aucun fichier écrit sur disque)."""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from exception.exceptions import InvalidImageError

# Photos iPhone (HEIC) acceptées si pillow-heif est installé
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


def decode_image(data: bytes, max_side: int, field: str | None = None) -> np.ndarray:
    """Octets -> image BGR, rotation EXIF appliquée, réduite à max_side.

    Le contenu est réellement décodé : un fichier renommé en .jpg est refusé.
    """
    if not data:
        raise InvalidImageError("Le fichier est vide.", field)
    try:
        with Image.open(io.BytesIO(data)) as pil:
            pil = ImageOps.exif_transpose(pil).convert("RGB")
            img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            "Le fichier n'est pas une image valide (JPEG, PNG, WebP ou HEIC).", field
        ) from exc

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def load_image(path: str, max_side: int) -> np.ndarray:
    with open(path, "rb") as f:
        return decode_image(f.read(), max_side)