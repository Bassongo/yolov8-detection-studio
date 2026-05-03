"""
Dashboard - Detection d'Objets avec YOLOv8
===========================================
Projet Deep Learning - AS3

Lancer avec : python app.py
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
import tempfile

# --- Chargement des modeles ---
MODELS = {}

def charger_modeles():
    MODELS["YOLOv8n (pre-entraine)"] = YOLO("yolov8n.pt")
    print("[OK] YOLOv8n pre-entraine charge")
    if os.path.exists("best.pt"):
        MODELS["YOLOv8n (fine-tune)"] = YOLO("best.pt")
        print("[OK] Modele fine-tune charge")

charger_modeles()


# --- Helpers UI ---

PALETTE = {
    "ink": "#18181b",
    "ink_soft": "#3f3f46",
    "muted": "#71717a",
    "line": "#e4e4e7",
    "surface": "#ffffff",
    "bg": "#faf8f3",
    "teal": "#0d9488",
    "teal_soft": "#ccfbf1",
    "orange": "#ea580c",
    "orange_soft": "#ffedd5",
    "amber": "#b45309",
}

def carte_kpi(valeur, libelle, accent="teal"):
    couleur = PALETTE[accent]
    return f"""
    <div style="background:{PALETTE['surface']}; border:1px solid {PALETTE['line']}; border-radius:14px; padding:18px 20px; flex:1; min-width:120px;">
        <div style="font-family:'Fraunces',serif; font-size:30px; font-weight:600; color:{couleur}; line-height:1;">{valeur}</div>
        <div style="font-size:11px; color:{PALETTE['muted']}; text-transform:uppercase; letter-spacing:1.4px; margin-top:8px; font-weight:500;">{libelle}</div>
    </div>
    """

def barre_classe(nom, count, total):
    pct = count / total * 100 if total else 0
    return f"""
    <div style="display:flex; align-items:center; margin-bottom:10px;">
        <span style="width:110px; font-size:13px; font-weight:500; color:{PALETTE['ink_soft']};">{nom}</span>
        <div style="flex:1; background:{PALETTE['bg']}; border-radius:999px; height:8px; margin:0 12px; overflow:hidden;">
            <div style="width:{pct}%; background:linear-gradient(90deg,{PALETTE['teal']},{PALETTE['orange']}); height:100%; border-radius:999px;"></div>
        </div>
        <span style="font-size:13px; font-weight:600; color:{PALETTE['ink']}; min-width:24px; text-align:right;">{count}</span>
    </div>
    """


# --- Fonctions de detection ---

def detecter_image(image, modele_choisi, seuil_confiance, seuil_iou):
    if image is None:
        return None, f"<p style='color:{PALETTE['muted']}; padding:12px;'>Uploadez une image pour commencer.</p>"

    model = MODELS.get(modele_choisi)
    if model is None:
        return None, f"<p style='color:{PALETTE['orange']}; padding:12px;'>Modele non disponible.</p>"

    start = time.time()
    results = model.predict(source=image, conf=seuil_confiance, iou=seuil_iou, verbose=False)
    inference_time = (time.time() - start) * 1000

    r = results[0]
    annotated = r.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    n_objets = len(r.boxes)
    if n_objets > 0:
        classes_detectees = {}
        conf_totale = 0
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            conf_totale += conf
            classes_detectees[cls_name] = classes_detectees.get(cls_name, 0) + 1

        conf_moyenne = conf_totale / n_objets

        kpis = (
            carte_kpi(n_objets, "Objets", "teal")
            + carte_kpi(f"{conf_moyenne:.0%}", "Confiance", "orange")
            + carte_kpi(f"{inference_time:.0f}ms", "Inference", "amber")
        )

        barres = ""
        for cls, count in sorted(classes_detectees.items(), key=lambda x: -x[1]):
            barres += barre_classe(cls, count, n_objets)

        stats = f"""
<div style="padding:8px 0;">
    <div style="display:flex; gap:12px; margin-bottom:18px; flex-wrap:wrap;">{kpis}</div>
    <div style="background:{PALETTE['surface']}; border:1px solid {PALETTE['line']}; border-radius:14px; padding:18px 20px;">
        <div style="font-weight:600; margin-bottom:14px; color:{PALETTE['ink']}; font-size:12px; text-transform:uppercase; letter-spacing:1.4px;">Classes detectees</div>
        {barres}
    </div>
</div>"""
    else:
        stats = f"""
<div style="padding:28px; text-align:center; background:{PALETTE['surface']}; border:1px dashed {PALETTE['line']}; border-radius:14px;">
    <div style="font-family:'Fraunces',serif; font-size:18px; color:{PALETTE['ink']};">Aucun objet detecte</div>
    <div style="font-size:12px; color:{PALETTE['muted']}; margin-top:6px;">Seuil : {seuil_confiance:.0%} - Inference : {inference_time:.0f}ms</div>
</div>"""

    return annotated_rgb, stats


def detecter_webcam(frame, modele_choisi, seuil_confiance):
    if frame is None:
        return None

    model = MODELS.get(modele_choisi)
    if model is None:
        return frame

    results = model.predict(source=frame, conf=seuil_confiance, iou=0.45, verbose=False)
    r = results[0]
    annotated = r.plot()

    n_det = len(r.boxes)
    cv2.putText(annotated, f"Objets: {n_det}",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (13, 148, 136), 2)
    return annotated


def detecter_video(video_path, modele_choisi, seuil_confiance, progress=gr.Progress()):
    if video_path is None:
        return None, f"<p style='color:{PALETTE['muted']}; padding:12px;'>Uploadez une video pour commencer.</p>"

    model = MODELS.get(modele_choisi)
    if model is None:
        return None, f"<p style='color:{PALETTE['orange']}; padding:12px;'>Modele non disponible.</p>"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"<p style='color:{PALETTE['orange']}; padding:12px;'>Impossible d'ouvrir la video.</p>"

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    total_detections = 0
    all_classes = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        progress(frame_count / max(total_frames, 1), desc=f"Frame {frame_count}/{total_frames}")

        results = model.predict(frame, conf=seuil_confiance, verbose=False)
        r = results[0]
        annotated = r.plot()

        n_det = len(r.boxes)
        total_detections += n_det
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            all_classes[cls_name] = all_classes.get(cls_name, 0) + 1

        cv2.putText(annotated, f"Frame {frame_count}/{total_frames} - Objets {n_det}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (13, 148, 136), 2)
        out.write(annotated)

    cap.release()
    out.release()

    avg_det = total_detections / max(frame_count, 1)

    kpis = (
        carte_kpi(frame_count, "Frames", "teal")
        + carte_kpi(total_detections, "Detections", "orange")
        + carte_kpi(f"{avg_det:.1f}", "Moy/frame", "amber")
    )

    total_cls = sum(all_classes.values()) or 1
    barres = ""
    for cls, count in sorted(all_classes.items(), key=lambda x: -x[1]):
        barres += barre_classe(cls, count, total_cls)

    stats = f"""
<div style="padding:8px 0;">
    <div style="display:flex; gap:12px; margin-bottom:18px; flex-wrap:wrap;">{kpis}</div>
    <div style="background:{PALETTE['surface']}; border:1px solid {PALETTE['line']}; border-radius:14px; padding:18px 20px;">
        <div style="font-weight:600; margin-bottom:14px; color:{PALETTE['ink']}; font-size:12px; text-transform:uppercase; letter-spacing:1.4px;">Classes detectees</div>
        {barres}
    </div>
</div>"""
    return output_path, stats


# --- Theme et CSS ---

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.zinc,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill=PALETTE["bg"],
    body_text_color=PALETTE["ink"],
    block_background_fill=PALETTE["surface"],
    block_border_width="1px",
    block_border_color=PALETTE["line"],
    block_shadow="0 1px 2px rgba(24,24,27,0.04)",
    block_radius="14px",
    block_title_text_color=PALETTE["ink"],
    block_title_text_weight="600",
    block_label_text_color=PALETTE["muted"],
    block_label_background_fill=PALETTE["surface"],
    button_primary_background_fill=PALETTE["ink"],
    button_primary_background_fill_hover=PALETTE["ink_soft"],
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_primary_shadow="0 1px 2px rgba(24,24,27,0.08)",
    button_secondary_background_fill=PALETTE["surface"],
    button_secondary_background_fill_hover=PALETTE["bg"],
    button_secondary_text_color=PALETTE["ink"],
    button_secondary_border_color=PALETTE["line"],
    input_border_color=PALETTE["line"],
    input_background_fill=PALETTE["surface"],
    input_shadow="none",
    slider_color=PALETTE["teal"],
)

custom_css = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&display=swap');

.gradio-container {{
    background: {PALETTE['bg']} !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}}

/* Hero */
.hero {{
    padding: 56px 24px 28px;
    text-align: center;
    border-bottom: 1px solid {PALETTE['line']};
    margin-bottom: 28px;
    position: relative;
}}
.hero .eyebrow {{
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {PALETTE['teal']};
    background: {PALETTE['teal_soft']};
    padding: 6px 14px;
    border-radius: 999px;
    margin-bottom: 22px;
}}
.hero h1 {{
    font-family: 'Fraunces', serif !important;
    font-size: 52px !important;
    font-weight: 600 !important;
    color: {PALETTE['ink']} !important;
    letter-spacing: -1.5px;
    line-height: 1.05 !important;
    margin: 0 0 14px !important;
}}
.hero h1 em {{
    font-style: italic;
    color: {PALETTE['orange']};
    font-weight: 500;
}}
.hero .lede {{
    color: {PALETTE['muted']};
    font-size: 16px;
    font-weight: 400;
    max-width: 560px;
    margin: 0 auto 28px;
    line-height: 1.55;
}}
.hero-stats {{
    display: flex;
    justify-content: center;
    gap: 36px;
    flex-wrap: wrap;
    margin-top: 16px;
}}
.hero-stats .stat {{
    text-align: center;
}}
.hero-stats .stat .num {{
    font-family: 'Fraunces', serif;
    font-size: 28px;
    font-weight: 600;
    color: {PALETTE['ink']};
    line-height: 1;
}}
.hero-stats .stat .lbl {{
    font-size: 11px;
    color: {PALETTE['muted']};
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-top: 6px;
    font-weight: 500;
}}

/* Tabs */
.tabs > .tab-nav {{
    background: {PALETTE['surface']} !important;
    border-radius: 999px !important;
    padding: 5px !important;
    margin: 0 auto 24px !important;
    border: 1px solid {PALETTE['line']} !important;
    display: inline-flex !important;
    width: fit-content !important;
    box-shadow: 0 1px 2px rgba(24,24,27,0.04) !important;
}}
.tabs > .tab-nav > button {{
    border-radius: 999px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 9px 22px !important;
    color: {PALETTE['muted']} !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.18s ease !important;
}}
.tabs > .tab-nav > button.selected {{
    background: {PALETTE['ink']} !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}}
.tabs > .tab-nav > button:hover:not(.selected) {{
    color: {PALETTE['ink']} !important;
    background: {PALETTE['bg']} !important;
}}

/* Section title */
.section-title {{
    font-family: 'Fraunces', serif;
    font-size: 22px;
    font-weight: 600;
    color: {PALETTE['ink']};
    margin: 8px 0 6px;
    letter-spacing: -0.4px;
}}
.section-sub {{
    color: {PALETTE['muted']};
    font-size: 14px;
    margin-bottom: 22px;
}}

/* Info box */
.info-box {{
    background: {PALETTE['orange_soft']};
    border: 1px solid #fed7aa;
    border-left: 3px solid {PALETTE['orange']};
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 18px;
    color: {PALETTE['ink_soft']};
    font-size: 13px;
    line-height: 1.55;
}}
.info-box strong {{
    color: {PALETTE['orange']};
    font-weight: 600;
}}

/* Cards a propos */
.about-card {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['line']};
    border-radius: 14px;
    padding: 24px;
    transition: all 0.2s ease;
}}
.about-card:hover {{
    border-color: {PALETTE['teal']};
    transform: translateY(-2px);
}}
.about-card h4 {{
    font-family: 'Fraunces', serif !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: {PALETTE['ink']} !important;
    margin-bottom: 8px !important;
    letter-spacing: -0.3px;
}}
.about-card p {{
    font-size: 13px;
    color: {PALETTE['muted']};
    line-height: 1.6;
}}

.tech-tag {{
    display: inline-block;
    background: {PALETTE['bg']};
    border: 1px solid {PALETTE['line']};
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
    color: {PALETTE['ink_soft']};
    margin: 3px;
}}

/* Footer */
.footer {{
    text-align: center;
    padding: 36px 14px 20px;
    color: {PALETTE['muted']};
    font-size: 12px;
    letter-spacing: 0.4px;
    border-top: 1px solid {PALETTE['line']};
    margin-top: 40px;
}}
.footer .dot {{
    color: {PALETTE['line']};
    margin: 0 10px;
}}

/* Sliders & labels */
.gradio-container label {{
    color: {PALETTE['ink_soft']} !important;
    font-weight: 500 !important;
}}

/* Primary button bigger */
button.primary, .gr-button-primary {{
    letter-spacing: 0.3px;
    font-weight: 600 !important;
}}

/* Mobile */
@media (max-width: 720px) {{
    .hero h1 {{ font-size: 36px !important; }}
    .hero {{ padding: 36px 16px 22px; }}
    .hero-stats {{ gap: 22px; }}
}}
"""


# --- Interface ---

modeles_dispo = list(MODELS.keys())
modele_default = modeles_dispo[0] if modeles_dispo else "YOLOv8n (pre-entraine)"
nb_classes_coco = 80

with gr.Blocks(title="YOLOv8 Detection Studio") as app:

    gr.HTML(f"""
    <div class="hero">
        <span class="eyebrow">Projet Deep Learning - AS3</span>
        <h1>Detection d'objets <em>en un clin d'oeil</em></h1>
        <div class="lede">
            Un studio interactif autour de YOLOv8 : image, webcam, video.
            Deux modeles compares, des metriques claires, une experience instantanee.
        </div>
        <div class="hero-stats">
            <div class="stat"><div class="num">{len(MODELS)}</div><div class="lbl">Modeles charges</div></div>
            <div class="stat"><div class="num">{nb_classes_coco}</div><div class="lbl">Classes COCO</div></div>
            <div class="stat"><div class="num">3</div><div class="lbl">Modes de detection</div></div>
        </div>
    </div>
    """)

    with gr.Tabs():

        # --- IMAGE ---
        with gr.TabItem("Image"):
            gr.HTML('<div class="section-title">Detection sur image</div><div class="section-sub">Glissez une image ou choisissez un exemple, ajustez les seuils, lancez l\'inference.</div>')

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    img_input = gr.Image(type="numpy", label="Image d'entree", height=380)

                    img_modele = gr.Dropdown(
                        choices=modeles_dispo, value=modele_default, label="Modele"
                    )
                    with gr.Row():
                        img_conf = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                            label="Confiance",
                        )
                        img_iou = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.45, step=0.05,
                            label="IoU (NMS)",
                        )
                    img_btn = gr.Button("Lancer la detection", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Image(label="Resultat", height=380)
                    img_stats = gr.HTML()

            img_btn.click(
                fn=detecter_image,
                inputs=[img_input, img_modele, img_conf, img_iou],
                outputs=[img_output, img_stats]
            )

        # --- WEBCAM ---
        with gr.TabItem("Webcam"):
            gr.HTML('<div class="section-title">Detection en temps reel</div><div class="section-sub">Activez votre camera et regardez le modele annoter le flux a la volee.</div>')
            gr.HTML("""
            <div class="info-box">
                Cliquez sur <strong>Enregistrer</strong> pour demarrer la detection.
                <strong>Arreter</strong> stoppe le flux. Necessite l'autorisation camera du navigateur.
            </div>
            """)

            with gr.Row():
                cam_modele = gr.Dropdown(
                    choices=modeles_dispo, value=modele_default, label="Modele", scale=2,
                )
                cam_conf = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                    label="Seuil de confiance", scale=1,
                )

            with gr.Row(equal_height=True):
                with gr.Column():
                    cam_input = gr.Image(
                        sources=["webcam"], streaming=True, type="numpy",
                        label="Webcam", height=420,
                    )
                with gr.Column():
                    cam_output = gr.Image(label="Detection", height=420)

            cam_input.stream(
                fn=detecter_webcam,
                inputs=[cam_input, cam_modele, cam_conf],
                outputs=[cam_output],
                time_limit=300,
                stream_every=0.15,
                concurrency_limit=30,
            )

        # --- VIDEO ---
        with gr.TabItem("Video"):
            gr.HTML('<div class="section-title">Detection sur video</div><div class="section-sub">Traitement frame par frame avec barre de progression et statistiques globales.</div>')

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    vid_input = gr.Video(label="Video d'entree")
                    vid_modele = gr.Dropdown(
                        choices=modeles_dispo, value=modele_default, label="Modele"
                    )
                    vid_conf = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                        label="Seuil de confiance"
                    )
                    vid_btn = gr.Button("Traiter la video", variant="primary", size="lg")

                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Video traitee")
                    vid_stats = gr.HTML()

            vid_btn.click(
                fn=detecter_video,
                inputs=[vid_input, vid_modele, vid_conf],
                outputs=[vid_output, vid_stats]
            )

        # --- A PROPOS ---
        with gr.TabItem("A propos"):
            gr.HTML('<div class="section-title">A propos du projet</div><div class="section-sub">Stack technique, architecture du modele et references academiques.</div>')

            gr.HTML(f"""
            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin-bottom:20px;">
                <div class="about-card">
                    <h4>Detection sur image</h4>
                    <p>Uploadez une image et obtenez les objets detectes avec leurs scores de confiance et leurs positions.</p>
                </div>
                <div class="about-card">
                    <h4>Webcam temps reel</h4>
                    <p>Activez votre webcam pour une detection continue directement dans le navigateur, sans upload.</p>
                </div>
                <div class="about-card">
                    <h4>Analyse video</h4>
                    <p>Traitez vos fichiers video et obtenez des statistiques detaillees par classe et par frame.</p>
                </div>
            </div>

            <div class="about-card" style="margin-bottom:16px;">
                <h4 style="margin-bottom:14px !important;">Technologies</h4>
                <div>
                    <span class="tech-tag">PyTorch</span>
                    <span class="tech-tag">YOLOv8</span>
                    <span class="tech-tag">Ultralytics</span>
                    <span class="tech-tag">Gradio</span>
                    <span class="tech-tag">OpenCV</span>
                    <span class="tech-tag">Python</span>
                    <span class="tech-tag">COCO Dataset</span>
                </div>
            </div>

            <div class="about-card" style="margin-bottom:16px;">
                <h4 style="margin-bottom:16px !important;">Architecture YOLOv8</h4>
                <div style="display:flex; align-items:center; justify-content:center; gap:10px; flex-wrap:wrap; padding:8px 0;">
                    <div style="background:{PALETTE['bg']}; border:1px solid {PALETTE['teal']}; padding:14px 22px; border-radius:12px; text-align:center;">
                        <div style="font-weight:600; font-size:14px; color:{PALETTE['teal']};">Backbone</div>
                        <div style="font-size:11px; color:{PALETTE['muted']}; margin-top:2px;">CSPDarknet</div>
                    </div>
                    <div style="font-size:18px; color:{PALETTE['muted']};">&rarr;</div>
                    <div style="background:{PALETTE['bg']}; border:1px solid {PALETTE['orange']}; padding:14px 22px; border-radius:12px; text-align:center;">
                        <div style="font-weight:600; font-size:14px; color:{PALETTE['orange']};">Neck</div>
                        <div style="font-size:11px; color:{PALETTE['muted']}; margin-top:2px;">FPN + PAN</div>
                    </div>
                    <div style="font-size:18px; color:{PALETTE['muted']};">&rarr;</div>
                    <div style="background:{PALETTE['bg']}; border:1px solid {PALETTE['amber']}; padding:14px 22px; border-radius:12px; text-align:center;">
                        <div style="font-weight:600; font-size:14px; color:{PALETTE['amber']};">Head</div>
                        <div style="font-size:11px; color:{PALETTE['muted']}; margin-top:2px;">Anchor-Free</div>
                    </div>
                </div>
            </div>

            <div class="about-card">
                <h4 style="margin-bottom:12px !important;">References</h4>
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <a href="https://docs.ultralytics.com/" target="_blank" style="color:{PALETTE['teal']}; text-decoration:none; font-size:13px; font-weight:500;">Ultralytics YOLOv8 Documentation</a>
                    <a href="https://cocodataset.org/" target="_blank" style="color:{PALETTE['teal']}; text-decoration:none; font-size:13px; font-weight:500;">COCO Dataset</a>
                    <a href="https://arxiv.org/abs/1506.02640" target="_blank" style="color:{PALETTE['teal']}; text-decoration:none; font-size:13px; font-weight:500;">YOLO v1 - Redmon et al. (2016)</a>
                    <a href="https://arxiv.org/abs/1804.02767" target="_blank" style="color:{PALETTE['teal']}; text-decoration:none; font-size:13px; font-weight:500;">YOLOv3 - Redmon and Farhadi (2018)</a>
                </div>
            </div>
            """)

    gr.HTML("""
    <div class="footer">
        YOLOv8 Detection Studio
        <span class="dot">/</span>
        Projet Deep Learning AS3
        <span class="dot">/</span>
        Construit avec Gradio et Ultralytics
    </div>
    """)


# --- Lancement ---
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
        css=custom_css,
    )
