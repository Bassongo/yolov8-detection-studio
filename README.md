# YOLOv8 Detection Studio

> Interactive object detection studio built on YOLOv8: run detection on images, live webcam streams and video files from a single Gradio interface.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-00BFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-6.0+-FF7C00.svg)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**[Live demo](https://huggingface.co/spaces/bassongo/deepl)**

## Overview

This studio wraps a YOLOv8 detection pipeline in an interactive interface, so detection parameters can be adjusted and their effect observed immediately. It supports three input modes (image, webcam, video) and lets a pretrained model be compared side by side with a fine-tuned one.

## Features

- Image detection with adjustable confidence and IoU thresholds
- Real-time detection through the webcam
- Video processing with per-class detection statistics
- Side-by-side comparison of a pretrained model and a fine-tuned model

## Tech Stack

- PyTorch and Ultralytics YOLOv8
- Gradio (interface)
- OpenCV (video processing)
- COCO128 dataset

## Quick Start

```bash
git clone https://github.com/Bassongo/yolov8-detection-studio.git
cd yolov8-detection-studio
pip install -r requirements.txt
```

### Getting the model weights

The `yolov8n.pt` (pretrained) and `best.pt` (fine-tuned) weights are not tracked in the repository because of GitHub LFS limits. Retrieve them as follows:

```bash
# yolov8n.pt is downloaded automatically by Ultralytics on first run

# best.pt (fine-tuned), from the Hugging Face Space
curl -L -o best.pt https://huggingface.co/spaces/bassongo/deepl/resolve/main/best.pt
```

### Running the app

```bash
python app.py
```

The application opens at http://localhost:7860

## Project Structure

```
.
├── app.py              # Gradio dashboard
├── requirements.txt    # Python dependencies
└── README.md
```

## Presentation Outline

The work was structured around four topics:

1. Introduction to object detection
2. YOLOv8 architecture (CSPDarknet backbone, FPN and PAN neck, anchor-free head)
3. Training on an annotated dataset (COCO128)
4. Real-time demonstration through the interactive studio

## Team

Academic project for the Deep Learning course at the École Nationale de la Statistique et de l'Analyse Économique Pierre Ndiaye (ENSAE Dakar), AS3 class.

| Name | Role |
|---|---|
| Ndeye Aissatou Cisse | Student |
| Armand Djekonbe Ndoasnan | Student |
| **Marc Mare** | Student |
| Cheikh Oumar Sakho | Student |

Supervised by Mrs Fatou Sall, Statistician Economist Engineer (ISE).

## Author

**Marc Mare**, [LinkedIn](https://www.linkedin.com/in/marc-mare-4875a6277) · [GitHub](https://github.com/Bassongo)
ENSAE Dakar | MSc SEP, University of Reims (2026)

## License

MIT License, see [LICENSE](LICENSE) for details.
