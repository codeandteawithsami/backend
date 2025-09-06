import cv2
import os
from ultralytics import YOLO
from PIL import Image

def detect_data(image_path: str, output_dir: str = "outputs", model_path: str = "model/best.pt", conf_thresh: float = 0.5):
    """
    Run YOLO detection, save detection image and crop only the highest confidence object above threshold.

    Returns:
        dict: {
            "detection_image": str,
            "crop_image": str or None
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    results = model(image_path)
    img = cv2.imread(image_path)

    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf >= conf_thresh:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "cls_id": cls_id,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })

    if not detections:
        return {"detection_image": None, "crop_image": None}

    best_det = max(detections, key=lambda d: d["conf"])
    x1, y1, x2, y2 = best_det["bbox"]
    cls_id = best_det["cls_id"]
    conf = best_det["conf"]

    label = f"{model.names[cls_id]} {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    crop = Image.open(image_path).crop((x1, y1, x2, y2))
    crop_path = os.path.join(output_dir, f"crop_best_{model.names[cls_id]}.jpg")
    crop.save(crop_path)

    detection_path = os.path.join(output_dir, "detection.jpg")
    cv2.imwrite(detection_path, img)

    return {"detection_image": detection_path, "crop_image": crop_path}


# Example usage
if __name__ == "__main__":
    paths = detect_data("data/image (1).png", output_dir="results")
    print(paths)
