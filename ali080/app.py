import os
import glob
import gradio as gr
from ultralytics import YOLO

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Download best.pt (see README) "
        "and place it in the 'models' folder."
    )

model = YOLO(MODEL_PATH)
CLASS_CHOICES = ["All", "With Helmet", "Without Helmet"]


THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "sans-serif"],
)

HEADER_GRADIENT = "linear-gradient(90deg, #f97316, #fbbf24)"  # header colours

CSS = f"""
.gradio-container {{ max-width: 1150px !important; margin: auto !important; }}
.app-header {{
    background: {HEADER_GRADIENT};
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 12px;
}}
.app-header, .app-header * {{ color: #ffffff !important; }}
.app-header h1 {{ margin: 0 0 6px 0; font-size: 2rem; }}
.app-header p {{ margin: 0; font-size: 1.05rem; }}
.app-footer {{ text-align: center; opacity: 0.75; font-size: 0.9rem; }}
button.primary {{ font-weight: 600; letter-spacing: 0.3px; }}
"""


def make_banner(with_n, without_n, class_filter):
    """Build a coloured status message from the detection counts."""
    if class_filter != "All":
        text = f"Filter active: showing only '{class_filter}' detections."
        bg, fg = "#e2e3e5", "#383d41"
    elif without_n > 0:
        text = f"⚠ Warning: {without_n} rider(s) without a helmet detected."
        bg, fg = "#f8d7da", "#721c24"
    elif with_n > 0:
        text = f"✅ All {with_n} detected rider(s) are wearing a helmet."
        bg, fg = "#d4edda", "#155724"
    else:
        text = "No riders detected. Try lowering the confidence threshold."
        bg, fg = "#e2e3e5", "#383d41"
    return (
        f"<div style='padding:14px;border-radius:10px;font-size:18px;"
        f"font-weight:600;background:{bg};color:{fg};'>{text}</div>"
    )


def detect(image, confidence, class_filter, show_conf):
    if image is None:
        raise gr.Error("Please upload an image or take a photo first.")

    bgr = image[:, :, ::-1]  # Gradio gives RGB, YOLO expects BGR

    # Optional class filter: keep only the chosen class
    classes = None
    if class_filter != "All":
        classes = [i for i, name in model.names.items() if name == class_filter]

    result = model.predict(
        bgr, conf=confidence, imgsz=640, classes=classes, verbose=False
    )[0]
    annotated = result.plot(conf=show_conf)[:, :, ::-1]  # BGR -> RGB for display

    rows = []
    with_n = without_n = 0
    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        score = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        rows.append([name, round(score, 2), f"({x1}, {y1}) to ({x2}, {y2})"])
        if name == "With Helmet":
            with_n += 1
        elif name == "Without Helmet":
            without_n += 1

    rows.sort(key=lambda r: r[1], reverse=True)
    return make_banner(with_n, without_n, class_filter), annotated, rows


example_paths = sorted(
    glob.glob(os.path.join(BASE_DIR, "examples", "*.jpg"))
    + glob.glob(os.path.join(BASE_DIR, "examples", "*.png"))
)

with gr.Blocks(title="Helmet Detection") as demo:
    gr.Markdown(
        "# 🪖 Helmet Detection\n"
        "Upload a photo or use your webcam. The model finds each rider and "
        "tells you whether they are wearing a helmet.",
        elem_classes=["app-header"],
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="numpy",
                sources=["upload", "webcam", "clipboard"],
                label="Image or webcam",
            )
            conf_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                    label="Confidence threshold")
            class_filter = gr.Dropdown(CLASS_CHOICES, value="All",
                                       label="Show classes")
            show_conf = gr.Checkbox(value=True, label="Show confidence scores")
            detect_btn = gr.Button("Detect", variant="primary", size="lg")
        with gr.Column():
            banner = gr.HTML()
            output_image = gr.Image(label="Detections")
            table = gr.Dataframe(
                headers=["Class", "Confidence", "Box (x1, y1 to x2, y2)"],
                interactive=False,
                label="Detected objects",
            )

    if example_paths:
        gr.Examples(
            examples=[[p, 0.4, "All", True] for p in example_paths],
            inputs=[input_image, conf_slider, class_filter, show_conf],
        )

    detect_btn.click(
        detect,
        [input_image, conf_slider, class_filter, show_conf],
        [banner, output_image, table],
    )

    gr.Markdown(
        "Model: YOLOv8n fine-tuned on the Bike Helmet Detection dataset "
        "(Roboflow Universe, Public Domain). Classes: With Helmet, Without Helmet.",
        elem_classes=["app-footer"],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, inbrowser=True)