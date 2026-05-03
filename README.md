# YOLOv8 Detection Studio

Studio interactif autour de YOLOv8 : detection d'objets sur image, webcam et video.

**Demo en ligne :** https://huggingface.co/spaces/bassongo/deepl

## Fonctionnalites

- Detection sur image avec choix du seuil de confiance et IoU
- Detection en temps reel via webcam
- Traitement de video avec statistiques par classe
- Comparaison entre modele pre-entraine et fine-tune

## Stack

- PyTorch + Ultralytics YOLOv8
- Gradio (interface)
- OpenCV (traitement video)
- Dataset COCO 128

## Installation locale

```bash
git clone https://github.com/Bassongo/yolov8-detection-studio.git
cd yolov8-detection-studio
pip install -r requirements.txt
```

### Recuperer les modeles

Les poids `yolov8n.pt` (pre-entraine) et `best.pt` (fine-tune) ne sont pas inclus dans le repo (limite LFS GitHub). Recupere-les depuis le Hugging Face Space :

```bash
# yolov8n.pt sera telecharge automatiquement par Ultralytics au premier lancement

# best.pt (fine-tune) - depuis le Space HF
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
├── requirements.txt    # Dependances Python
└── README.md
```

## Plan d'expose (Projet Deep Learning - AS3)

1. Introduction a la detection d'objets
2. Architecture YOLOv8 (Backbone CSPDarknet, Neck FPN+PAN, Head Anchor-Free)
3. Entrainement sur dataset annote (COCO128)
4. Demonstration temps reel via le studio interactif

## License

MIT
