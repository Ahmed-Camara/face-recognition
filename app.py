"""API de vérification d'identité : selfie vs pièce d'identité (InsightFace).

POST /compare_photo   multipart/form-data, champs : id_card, selfie
GET  /health
"""
import logging
import time

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from exception.exceptions import FaceVerificationError
from src.face_engine import get_engine
from src.verifier import verify_selfie_against_id
from utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

cfg = load_config()
api_cfg = cfg["api"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(api_cfg["max_upload_mb"]) * 1024 * 1024
CORS(app, resources={r"/compare_photo": {"origins": api_cfg["cors_origins"]}})

FIELDS = {"id_card": "la pièce d'identité", "selfie": "le selfie"}


def _read_upload(name: str) -> bytes | None:
    f = request.files.get(name)
    if f is None or not f.filename:
        return None
    return f.read()


@app.post("/compare_photo")
def compare_photo():
    response = {}
    data = {name: _read_upload(name) for name in FIELDS}
    missing = [name for name, content in data.items() if content is None]
    if missing:
        return jsonify({
            "message": "Envoyez les deux photos : champs 'id_card' (pièce d'identité) et 'selfie'.",
            "statut": "0",
            "code": "missing_file",
            "champs_manquants": missing,
        }), 400

    start = time.perf_counter()
    try:
        result = verify_selfie_against_id(data["id_card"], data["selfie"])
        response["decision"] = result["decision"]
        response["match"] = result["match"]
        response["similarity"] = result["similarity"]
        response["message"] = ""
        
        if result["match"]:
            result["statut"] = "1"
        else:
            result["statut"] = "0"
        print(result)
    except FaceVerificationError as exc:
        log.info("Refus métier : %s (%s)", exc.code, exc.field)
        return jsonify({"message": str(exc), "code": exc.code, "champ": exc.field}), 422
        #response["message"] = str(exc)
        #response["statut"] = "0"
        #return jsonify(response), 422
    except Exception:
        log.exception("Erreur interne pendant la vérification")
        return jsonify({"message": "Erreur interne du serveur.", "code": "internal_error"}), 500

    result["duree_ms"] = int((time.perf_counter() - start) * 1000)
    # Aucune donnée biométrique ni image n'est journalisée ou conservée
    log.info("Vérification : %s (similarité %.3f, %d ms)",
             result["decision"], result["similarity"], result["duree_ms"])
    return jsonify(result), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify({
        "message": f"Fichiers trop volumineux (max {api_cfg['max_upload_mb']} Mo au total).",
        "code": "too_large",
    }), 413


# Chargement des modèles au démarrage : la première requête n'est pas pénalisée
get_engine()

if __name__ == "__main__":
    from waitress import serve

    log.info("API sur http://%s:%s", api_cfg["host"], api_cfg["port"])
    serve(app, host=api_cfg["host"], port=int(api_cfg["port"]), threads=int(api_cfg["threads"]))