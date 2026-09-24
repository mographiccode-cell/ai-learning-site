from ultralytics import YOLO

# Transfer learning from official YOLOv8n pretrained weights.
# The source project trained a YOLOv8 model on the same 8 car-damage classes.
model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    project="runs",
    name="car_damage_yolov8n",
    pretrained=True,
    plots=True,
    seed=42,
)

print("Training complete.")
print("Best weights: runs/car_damage_yolov8n/weights/best.pt")
