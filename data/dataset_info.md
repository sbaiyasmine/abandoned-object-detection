# Dataset — Équipe 06 : Détection d'objets abandonnés

## Source
COCO

## Description
Dataset utilisé pour l'entraînement et l'évaluation du modèle MOG2 + YOLOv8n.

## Pré-traitement
- Redimensionnement : 640x640
- Normalisation : [0, 1]
- Format : NCHW float32

## Métriques cibles
- Détection objet abandonné > 85%
- Délai alarme < 5s
- FPS > 8 sur RPI4
