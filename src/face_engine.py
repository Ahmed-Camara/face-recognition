"""Moteur InsightFace : détection (SCRFD), vivacité, embedding ArcFace (512-d)."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis

from exception.exceptions import LowQualityFaceError, NoFaceFoundError, SpoofDetectedError
from utils.config import load_config

log = logging.getLogger(__name__)

_ROTATIONS = [
    (None, 0),
    (cv2.ROTATE_90_CLOCKWISE, 90),
    (cv2.ROTATE_180, 180),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, 270),
]


@dataclass
class FaceInfo:
    embedding: np.ndarray            # vecteur 512-d normalisé
    det_score: float
    face_px: int                     # largeur du visage dans l'image analysée
    rotation: int                    # rotation appliquée pour redresser l'image
    liveness: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def public(self) -> dict:
        """Infos renvoyées au client (jamais l'embedding)."""
        out = {
            "det_score": round(self.det_score, 3),
            "face_px": self.face_px,
            "rotation": self.rotation,
        }
        if self.liveness is not None:
            out["liveness"] = self.liveness
        return out


def _area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return float((x2 - x1) * (y2 - y1))


def _width(face) -> int:
    return int(face.bbox[2] - face.bbox[0])


def _is_upright(face) -> bool:
    """Yeux au-dessus du nez, nez au-dessus de la bouche : visage à l'endroit."""
    if face.kps is None:
        return True
    eyes_y = (face.kps[0][1] + face.kps[1][1]) / 2
    nose_y = face.kps[2][1]
    mouth_y = (face.kps[3][1] + face.kps[4][1]) / 2
    return eyes_y < nose_y < mouth_y


class FaceEngine:
    def __init__(self, cfg: dict):
        m, q, lv = cfg["model"], cfg["quality"], cfg.get("liveness", {})
        self.min_face_px = int(q["min_face_px"])
        self.upscale_below = int(q["upscale_below_px"])
        self.liveness_enabled = bool(lv.get("enabled", False))

        available = set(onnxruntime.get_available_providers())
        providers = [p for p in m["providers"] if p in available] or ["CPUExecutionProvider"]
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

        self._app = FaceAnalysis(
            name=m["pack"],
            allowed_modules=["detection", "recognition"],
            addons=["liveness"] if self.liveness_enabled else None,
            # "observe" : la vivacité est mesurée mais l'embedding est toujours calculé.
            # On ne l'exige que pour le selfie (la photo de la pièce est, par nature, imprimée).
            liveness_mode="observe",
            liveness_threshold=float(lv.get("threshold", 0.8)),
            providers=providers,
        )
        ds = int(m["det_size"])
        self._app.prepare(ctx_id=ctx_id, det_thresh=float(m["min_det_score"]), det_size=(ds, ds))
        self._lock = threading.Lock()
        log.info("Moteur prêt : %s sur %s, vivacité=%s", m["pack"], providers[0], self.liveness_enabled)

    # ------------------------------------------------------------------ interne
    def _detect(self, img: np.ndarray) -> list:
        with self._lock:
            return self._app.get(img)

    def _find(self, img: np.ndarray):
        """Cherche un visage à l'endroit, en tournant l'image si besoin."""
        for rot, deg in _ROTATIONS:
            cand = img if rot is None else cv2.rotate(img, rot)
            faces = [f for f in self._detect(cand) if _is_upright(f)]
            if faces:
                return faces, cand, deg
        # Visage recadré trop serré (bords coupés) : on élargit les bords et on réessaie
        h, w = img.shape[:2]
        pad = max(h, w) // 2
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        faces = [f for f in self._detect(padded) if _is_upright(f)]
        return faces, padded, 0

    def _refine_small(self, img: np.ndarray, face):
        """Petit visage (photo de pièce) : recadre autour, agrandit, recalcule.

        Les points du visage sont alors plus précis, donc l'alignement et
        l'embedding aussi.
        """
        x1, y1, x2, y2 = face.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * 1.2
        H, W = img.shape[:2]
        crop = img[int(max(0, cy - half)):int(min(H, cy + half)),
                   int(max(0, cx - half)):int(min(W, cx + half))]
        scale = 224 / max(x2 - x1, y2 - y1)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        faces = [f for f in self._detect(crop) if _is_upright(f)]
        return max(faces, key=_area) if faces else face

    # ----------------------------------------------------------------- public
    def extract(self, img: np.ndarray, field: str, label: str, require_live: bool = False) -> FaceInfo:
        faces, cand, rotation = self._find(img)
        if not faces:
            raise NoFaceFoundError(
                f"Aucun visage détecté sur {label}. Vérifiez que la photo est nette, "
                "bien éclairée et que le visage est entièrement visible.",
                field,
            )

        faces.sort(key=_area, reverse=True)
        face = faces[0]  # le plus grand (écarte la petite photo fantôme des CNI)
        width = _width(face)
        if width < self.min_face_px:
            raise LowQualityFaceError(
                f"Le visage sur {label} est trop petit ({width} px). "
                "Rapprochez l'appareil ou utilisez une photo de meilleure résolution.",
                field,
            )

        warnings = []
        if len(faces) > 1 and _area(faces[1]) >= 0.5 * _area(face):
            warnings.append(f"Plusieurs visages de taille comparable sur {label} : le plus grand a été retenu.")
        if width < self.upscale_below:
            face = self._refine_small(cand, face)

        liveness = None
        if self.liveness_enabled and require_live:
            liveness = dict(face.liveness)
            if liveness.get("status") != "ok":
                raise LowQualityFaceError(
                    f"Le visage sur {label} est trop près du bord pour le contrôle anti-fraude. "
                    "Centrez le visage et reculez légèrement.",
                    field,
                )
            if not liveness.get("is_live"):
                raise SpoofDetectedError(
                    f"{label.capitalize()} ne semble pas être une photo prise en direct "
                    "(photo d'écran ou impression). Prenez un nouveau selfie.",
                    field,
                )
            liveness["live_score"] = round(float(liveness["live_score"]), 3)
            liveness.pop("reason", None)

        return FaceInfo(
            embedding=face.normed_embedding,
            det_score=float(face.det_score),
            face_px=width,
            rotation=rotation,
            liveness=liveness,
            warnings=warnings,
        )


@lru_cache(maxsize=1)
def get_engine() -> FaceEngine:
    """Instance unique partagée (modèles chargés une seule fois)."""
    return FaceEngine(load_config())