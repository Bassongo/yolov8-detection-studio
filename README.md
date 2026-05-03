# YOLOv8 Detection Studio

Studio interactif autour de YOLOv8 : détection d'objets sur image, webcam et vidéo.

**Démo en ligne :** https://huggingface.co/spaces/bassongo/deepl

## Fonctionnalités

- Détection sur image avec choix du seuil de confiance et IoU
- Détection en temps réel via webcam
- Traitement de vidéo avec statistiques par classe
- Comparaison entre un modèle pré-entraîné et un modèle fine-tuné

## Stack

- PyTorch + Ultralytics YOLOv8
- Gradio (interface)
- OpenCV (traitement vidéo)
- Dataset COCO 128

## Installation locale

```bash
git clone https://github.com/Bassongo/yolov8-detection-studio.git
cd yolov8-detection-studio
pip install -r requirements.txt
```

### Récupérer les modèles

Les poids `yolov8n.pt` (pré-entraîné) et `best.pt` (fine-tuné) ne sont pas inclus dans le dépôt (limite LFS GitHub). Vous pouvez les récupérer ainsi :

```bash
# yolov8n.pt sera téléchargé automatiquement par Ultralytics au premier lancement

# best.pt (fine-tuné) - depuis le Space Hugging Face
curl -L -o best.pt https://huggingface.co/spaces/bassongo/deepl/resolve/main/best.pt
```

### Lancement

```bash
python app.py
```

L'application s'ouvre sur http://localhost:7860

## Structure du projet

```
.
├── app.py              # Dashboard Gradio
├── requirements.txt    # Dépendances Python
└── README.md
```

## Plan d'exposé (Projet Deep Learning - AS3)

Nous avons structuré notre travail autour des quatre axes suivants :

1. Introduction à la détection d'objets
2. Architecture YOLOv8 (Backbone CSPDarknet, Neck FPN+PAN, Head Anchor-Free)
3. Entraînement sur dataset annoté (COCO128)
4. Démonstration temps réel via le studio interactif

## Équipe

Projet réalisé dans le cadre du cours de Deep Learning à l'École Nationale de la Statistique et de l'Analyse Économique Pierre NDIAYE (ENSAE), promotion AS3.

Membres du groupe :
- Ndeye Aissatou CISSE
- Armand Djekonbe NDOASNAN
- Marc MARE
- Cheikh Oumar SAKHO

Sous la supervision de Mme Fatou SALL, Ingénieur Statisticien Économiste (ISE).

## Licence

MIT
