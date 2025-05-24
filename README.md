# Track4
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Charger le modèle YOLOv11 pré-entraîné
model = YOLO('best.pt')

# Initialiser le tracker DeepSORT
tracker = DeepSort(max_age=30)
from collections import defaultdict

# Dictionnaire pour stocker les temps d'apparition des objets
object_times = defaultdict(lambda: {"start": None, "end": None})
import cv2

# Ouvrir le fichier vidéo
video_path = "D:\VIDEON\VIDEON\DSC_8488.MOV"
cap = cv2.VideoCapture(video_path)

# Obtenir le nombre de frames par seconde (fps) de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialiser le compteur de frames
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Exécuter la détection avec YOLOv11 sur la frame actuelle
    results = model(frame)

    # Préparer les détections pour DeepSORT
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id))

    # Mettre à jour le tracker avec les détections actuelles
    tracks = tracker.update_tracks(detections, frame=frame)

    # Enregistrer les temps d'apparition des objets
    current_time = frame_idx / fps
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        if object_times[track_id]["start"] is None:
            object_times[track_id]["start"] = current_time
        object_times[track_id]["end"] = current_time

    # Incrémenter le compteur de frames
    frame_idx += 1

cap.release()
# Calculer la durée d'apparition de chaque objet
for track_id, times in object_times.items():
    if times["start"] is not None and times["end"] is not None:
        duration = times["end"] - times["start"]
        print(f"Objet ID {track_id}: Durée d'apparition = {duration:.2f} secondes")
