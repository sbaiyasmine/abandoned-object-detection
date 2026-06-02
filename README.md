# Détection d’Objets Abandonnés avec Vision Embarquée et Intelligence Artificielle

## Description du projet

Ce projet consiste à développer un système embarqué intelligent capable de détecter automatiquement des objets abandonnés à partir d’un flux vidéo temps réel sur Raspberry Pi 4.

Le système utilise :

- un modèle YOLOv8n fine-tuné ;
- ONNX Runtime pour l’inférence optimisée ;
- Picamera2 pour l’acquisition vidéo ;
- MQTT pour la communication temps réel ;
- des logs CSV pour la supervision et la traçabilité.

Le pipeline complet permet :

- l’acquisition des images ;
- le prétraitement des frames ;
- l’inférence du modèle ;
- la détection des objets ;
- l’analyse temporelle ;
- la génération automatique d’alertes ;
- la publication MQTT ;
- l’enregistrement des événements.

---

# Objectif du projet

Le système doit respecter les contraintes suivantes :

| Métrique | Objectif |
|---|---|
| Détection objet abandonné | > 85% |
| Délai d’alerte | < 5 secondes |
| FPS système | > 8 FPS |
| Communication MQTT | Temps réel |

---

# Technologies utilisées

- Python
- OpenCV
- YOLOv8n
- ONNX Runtime
- MQTT
- NumPy
- Picamera2
- Pytest
- Raspberry Pi 4

---

# Architecture du projet

```bash
equipe_06_objet-abandonne/
│
├── src/
│   ├── main.py
│   ├── pipeline.py
│   ├── model.py
│   └── mqtt_publisher.py
│
├── common/
│   └── base_vision.py
│
├── tests/
│   ├── test_pipeline.py
│   └── test_mqtt.py
│
├── models/
│   ├── yolov8n_finetuned.onnx
│   └── yolov8n_finetuned_320.onnx
│
├── logs/
│
├── results/
│
├── data/
│
├── config.json
│
├── requirements.txt
│
└── README.md
```

---

# Dataset utilisé

Le modèle a été entraîné sur une version réduite du dataset COCO appelée :

```bash
mini_coco
```

Le mini dataset contient :

- les images d’entraînement ;
- les images de validation ;
- les annotations YOLO ;
- les classes importantes pour le projet.

L’objectif de cette réduction est :

- de diminuer le temps d’entraînement ;
- de réduire la consommation mémoire ;
- d’adapter le projet aux contraintes Google Colab et Raspberry Pi.

---

# Entraînement du modèle

Le fine-tuning du modèle YOLOv8n a été réalisé sur Google Colab avec Ultralytics.

Paramètres principaux :

```python
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=32
)
```

Après entraînement, le modèle obtenu :

```bash
best.pt
```

a été exporté au format ONNX.

---

# Optimisation ONNX

Le modèle a été converti au format ONNX afin d’améliorer les performances sur Raspberry Pi 4.

Export utilisé :

```python
model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True
)
```

Le projet utilise ensuite une version optimisée :

```bash
yolov8n_finetuned_320.onnx
```

Cette version réduit le coût de calcul et améliore les FPS.

---

# Pipeline de traitement

Le pipeline complet réalise :

1. Acquisition vidéo avec Picamera2 ;
2. Resize des images en 320x320 ;
3. Conversion RGB ;
4. Normalisation des pixels ;
5. Inférence ONNX Runtime ;
6. Décodage des sorties YOLO ;
7. Analyse temporelle ;
8. Génération des alertes MQTT ;
9. Sauvegarde des logs CSV.

---

# Détection d’objet abandonné

Le système surveille la durée de présence des objets détectés.

Lorsqu’un objet reste présent plus longtemps que :

```python
abandoned_seconds = 4
```

le système :

- considère l’objet comme abandonné ;
- génère une alerte MQTT ;
- sauvegarde un événement CSV ;
- affiche une bounding box rouge.

---

# Communication MQTT

Le système utilise MQTT pour publier les événements temps réel.

Topics utilisés :

```bash
surveillance/equipe_06/events
surveillance/equipe_06/alert
surveillance/equipe_06/heartbeat
```

Les messages publiés contiennent :

- le type d’événement ;
- la classe détectée ;
- le score de confiance ;
- le niveau d’alerte ;
- le timestamp.

---

# Logs CSV

Tous les événements sont enregistrés automatiquement dans des fichiers CSV.

Exemples d’événements :

```bash
object_present
abandoned_object
```

Les logs permettent :

- le debugging ;
- la supervision ;
- la traçabilité ;
- l’analyse hors ligne.

---

# Optimisation FPS

Les premiers tests en résolution :

```bash
640x640
```

donnaient seulement :

```bash
3 à 5 FPS
```

Pour améliorer les performances :

- réduction de résolution 320x320 ;
- export ONNX ;
- optimisation CPU ;
- réduction de fréquence d’inférence.

Le benchmark utilise :

```python
INFER_EVERY = 2
```

L’inférence est donc exécutée une frame sur deux afin de réduire la charge CPU.

---

# Résultats obtenus

## Pipeline complet

Commande :

```bash
python3 -m src.main
```

Résultat :

```bash
≈ 4 FPS
```

Ce pipeline inclut :

- MQTT ;
- logs CSV ;
- alertes ;
- post-traitement ;
- affichage complet.

---

## Benchmark optimisé

Commande :

```bash
python3 benchmark_pipeline_fps.py
```

Résultat :

```bash
FPS système optimisé = 16.1 FPS
```

Le benchmark mesure uniquement :

- caméra ;
- preprocess ;
- inférence ONNX ;
- calcul FPS.

La contrainte :

```bash
FPS > 8
```

est donc validée.

---

# Installation

## Cloner le projet

```bash
git clone <repo_url>
```

## Accéder au dossier

```bash
cd equipe_06_objet-abandonne
```

## Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# Exécution

## Lancer le système principal

```bash
python3 -m src.main
```

## Lancer le benchmark FPS

```bash
python3 benchmark_pipeline_fps.py
```

## Exécuter les tests

```bash
pytest tests/ -v
```

---

# Tests unitaires

Les tests Pytest permettent de vérifier :

- le preprocess ;
- les messages MQTT ;
- les alertes ;
- les topics ;
- la normalisation ;
- les payloads JSON.

Des mocks sont utilisés pour simuler :

- MQTT ;
- ONNX Runtime ;
- Picamera2.

---

# Plateforme utilisée

- Raspberry Pi 4
- Picamera2
- CPUExecutionProvider
- Linux embarqué



---

# Conclusion

Le projet développé permet de réaliser une solution embarquée intelligente de détection d’objets abandonnés fonctionnant en temps réel sur Raspberry Pi 4.

L’utilisation de :

- YOLOv8n fine-tuné ;
- ONNX Runtime ;
- MQTT ;
- Picamera2 ;

a permis d’obtenir un pipeline léger, modulaire et optimisé compatible avec les contraintes des systèmes embarqués.
