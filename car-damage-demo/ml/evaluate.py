from ultralytics import YOLO

MODEL_PATH = "runs/car_damage_yolov8n/weights/best.pt"

model = YOLO(MODEL_PATH)

metrics = model.val(
    data="data.yaml",
    split="test",
    imgsz=640,
    conf=0.25,
    iou=0.60,
    plots=True,
)

print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
print(f"mAP@50:    {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")

# Export the final trained detector for browser/mobile/backend deployment.
model.export(format="onnx", imgsz=640, simplify=True)
print("ONNX export complete.")
