# Entraînement YOLOv8 sur mini_coco + export ONNX
# À exécuter sur Google Colab

!pip install ultralytics -q

from google.colab import drive, files
drive.mount('/content/drive')

!unzip -q "/content/drive/MyDrive/mini_coco.zip" -d "/content/drive/MyDrive/"

from pathlib import Path
import json
import shutil
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO

BASE_DIR = Path("/content/drive/MyDrive/mini_coco")

TRAIN_IMG_DIR = BASE_DIR / "train2017"
VAL_IMG_DIR = BASE_DIR / "val2017"

ANN_TRAIN = BASE_DIR / "annotations" / "instances_train2017.json"
ANN_VAL = BASE_DIR / "annotations" / "instances_val2017.json"

YOLO_DIR = Path("/content/yolo_dataset")


def coco_to_yolo(ann_path, img_src_dir, split):
    with open(ann_path) as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_map = {c["id"]: i for i, c in enumerate(categories)}
    cat_names = [c["name"] for c in categories]
    images = {img["id"]: img for img in data["images"]}

    ann_by_image = defaultdict(list)

    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 0:
            ann_by_image[ann["image_id"]].append(ann)

    out_img_dir = YOLO_DIR / "images" / split
    out_lbl_dir = YOLO_DIR / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{split}] Conversion de {len(images)} images...")

    for img_id, img_info in tqdm(images.items()):
        file_name = img_info["file_name"]
        W, H = img_info["width"], img_info["height"]

        src = img_src_dir / file_name
        dst = out_img_dir / file_name

        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

        lbl_path = out_lbl_dir / (Path(file_name).stem + ".txt")

        lines = []

        for ann in ann_by_image[img_id]:
            cls = cat_id_map[ann["category_id"]]
            x, y, w, h = ann["bbox"]

            cx = max(0, min(1, (x + w / 2) / W))
            cy = max(0, min(1, (y + h / 2) / H))
            nw = max(0, min(1, w / W))
            nh = max(0, min(1, h / H))

            if nw > 0 and nh > 0:
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

    return cat_names


cat_names = coco_to_yolo(ANN_TRAIN, TRAIN_IMG_DIR, "train")
coco_to_yolo(ANN_VAL, VAL_IMG_DIR, "val")

print("Conversion COCO vers YOLO terminée.")

yaml_path = YOLO_DIR / "data.yaml"

yaml_content = f"""path: {YOLO_DIR}
train: images/train
val: images/val
nc: {len(cat_names)}
names: {cat_names}
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print("data.yaml créé :", yaml_path)

model = YOLO("yolov8n.pt")

results = model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    project="/content/runs",
    name="yolov8n_minicoco",
    lr0=0.001,
    patience=15,
    plots=True,
)

print("Entraînement terminé.")

best_model_path = "/content/runs/yolov8n_minicoco/weights/best.pt"

model = YOLO(best_model_path)

model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True
)

print("Export ONNX terminé.")

files.download("/content/runs/yolov8n_minicoco/weights/best.onnx")
