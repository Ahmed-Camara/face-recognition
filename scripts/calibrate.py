"""Calibre les seuils de décision sur VOS paires selfie / pièce d'identité.

Fichier CSV attendu (avec en-tête) :
    id_card,selfie,same
    data/p1_id.jpg,data/p1_selfie.jpg,1
    data/p1_id.jpg,data/p2_selfie.jpg,0
    ...
same = 1 si même personne, 0 sinon. Visez au moins 100 paires de chaque type.

Usage (depuis la racine du projet) :
    python -m scripts.calibrate paires.csv
"""
import csv
import sys

import numpy as np

from exception.exceptions import FaceVerificationError
from src.face_engine import get_engine
from utils.config import load_config
from utils.image_io import load_image


def main(csv_path: str) -> None:
    cfg = load_config()
    max_side = int(cfg["img"]["max_side"])
    engine = get_engine()

    genuine, impostor, failures = [], [], 0
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                a = engine.extract(load_image(row["id_card"], max_side), "id_card", "la pièce")
                b = engine.extract(load_image(row["selfie"], max_side), "selfie", "le selfie")
            except (FaceVerificationError, OSError) as exc:
                failures += 1
                print(f"[échec] {row['id_card']} / {row['selfie']} : {exc}")
                continue
            sim = float(np.dot(a.embedding, b.embedding))
            (genuine if row["same"].strip() == "1" else impostor).append(sim)

    if not genuine or not impostor:
        sys.exit("Il faut des paires 'same=1' ET 'same=0'.")
    g, i = np.array(genuine), np.array(impostor)
    print(f"\nPaires valides : {len(g)} même personne, {len(i)} différentes, {failures} échecs")
    print(f"Même personne : min {g.min():.3f}  moyenne {g.mean():.3f}")
    print(f"Différentes   : max {i.max():.3f}  moyenne {i.mean():.3f}\n")

    print(" seuil | faux acceptés (FAR) | faux rejets (FRR)")
    for t in np.arange(0.20, 0.61, 0.02):
        far = (i >= t).mean() * 100
        frr = (g < t).mean() * 100
        print(f" {t:.2f}  | {far:6.2f} %            | {frr:6.2f} %")

    # Seuil conseillé : le plus bas qui n'accepte aucun imposteur de l'échantillon
    safe = float(i.max()) + 0.01
    print(f"\nSeuil 'match' conseillé (0 faux accepté sur cet échantillon) : {safe:.2f}")
    print("Placez 'review_threshold' quelques points en dessous pour la zone de contrôle manuel.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])