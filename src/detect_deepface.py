import numpy as np
from deepface import DeepFace
import cv2 as cv
from utils.image_io import load_image
from exception.exceptions import NoFaceFoundError
import sys


def detect_deepface(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=True,
            align=False
        )
    except ValueError:
        return []
    boxes = []
    
    for f in faces:
        area = f["facial_area"]
        
        if f.get("confidence",1) < 0:
            continue
        boxes.append((area["x"], area["y"], area["w"], area["h"]))
    return boxes

def detect_opencv(img: np.ndarray) -> list[tuple[int,int,int,int]]:
    cascade = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")
    
    gray = cv.equalizeHist(cv.cvtColor(img,cv.COLOR_BGR2GRAY))
    min_size = max(40,min(img.shape[:2]) // 12)
    
    faces = cascade.detectMultiScale(
        gray,scaleFactor=1.1,minNeighbors=6,minSize=(min_size,min_size)
    )
    return [tuple(int(v) for v in f) for f in faces]

def detect_faces(img: np.ndarray, detector: str) -> list[tuple[int, int, int, int]]:
    if detector == "opencv":
        return detect_opencv(img)
    try:
        return detect_deepface(img)
    except ImportError:
        print("[info] DeepFace non installé, fallback sur OpenCV Haar.", file=sys.stderr)
        return detect_opencv(img)
    
def crop_with_margin(img: np.ndarray, box: tuple[int, int, int, int], margin: float) -> np.ndarray:
    
    #Agrandit la boîte du visage pour obtenir un portrait type photo d'identité.
    #margin = proportion de la taille du visage ajoutée de chaque côté.
    #Plus de marge en bas (épaules) qu'en haut.
    
    x, y, w, h = box
    H, W = img.shape[:2]
 
    x1 = int(x - w * margin)
    x2 = int(x + w + w * margin)
    y1 = int(y - h * margin)
    y2 = int(y + h + h * margin * 1.6)
 
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return img[y1:y2, x1:x2]


def extract_id_photo(image_path: str, margin: float = 0.5, detector: str = "retinaface") -> np.ndarray:
    #Retourne la photo (BGR) extraite de la pièce d'identité.
    img = load_image(image_path)
 
    rotations = [
        (None, "0°"),
        (cv.ROTATE_90_CLOCKWISE, "90°"),
        (cv.ROTATE_180, "180°"),
        (cv.ROTATE_90_COUNTERCLOCKWISE, "270°"),
    ]
 
    for rot, label in rotations:
        candidate = img if rot is None else cv.rotate(img, rot)
        boxes = detect_faces(candidate, detector)
        if boxes:
            if rot is not None:
                print(f"[info] Carte orientée de {label}, image redressée.", file=sys.stderr)
            largest = max(boxes, key=lambda b: b[2] * b[3])
            return crop_with_margin(candidate, largest, margin)
 
    raise NoFaceFoundError(
        "Aucun visage détecté sur la pièce d'identité. "
        "Vérifiez que l'image est nette, bien éclairée et que la photo est visible."
    )
