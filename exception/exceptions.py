class FaceVerificationError(Exception):
    """Erreur métier : renvoyée au client en 422 avec un message exploitable."""

    code = "verification_error"

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field  # "id_card" | "selfie" | None


class InvalidImageError(FaceVerificationError):
    """Fichier vide, corrompu ou qui n'est pas une image."""

    code = "invalid_image"


class NoFaceFoundError(FaceVerificationError):
    """Aucun visage détecté."""

    code = "no_face"


class LowQualityFaceError(FaceVerificationError):
    """Visage trop petit ou trop mal cadré pour être comparé."""

    code = "low_quality"


class SpoofDetectedError(FaceVerificationError):
    """Le selfie ressemble à une photo d'écran ou à une impression."""

    code = "spoof_detected"