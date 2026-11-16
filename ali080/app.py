import os
import gradio as gr
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Download best.pt (see README) "
        "and place it in the 'models' folder."
    )

model = YOLO(MODEL_PATH)


def detect(image, confidence):
    if image is None:
        raise gr.Error("Please upload an image first.")
    bgr = image[:, :, ::-1]  # Gradio gives RGB, YOLO expects BGR
    result = model.predict(bgr, conf=confidence, imgsz=640, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB for display

    counts = {}
    for cls_id in result.boxes.cls.tolist():
        name = model.names[int(cls_id)]
        counts[name] = counts.get(name, 0) + 1

    if not counts:
        summary = "No helmets or riders detected. Try lowering the confidence threshold."
    else:
        summary = "\n".join(f"{name}: {n}" for name, n in counts.items())
    return annotated, summary


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Upload an image"),
        gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Confidence threshold"),
    ],
    outputs=[
        gr.Image(label="Detections"),
        gr.Textbox(label="Summary"),
    ],
    title="Helmet Detection",
    description="Detects riders With Helmet and Without Helmet.",
)

if __name__ == "__main__":
    demo.launch()