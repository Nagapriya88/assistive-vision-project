import shutil, yaml, random
from pathlib import Path

COCO_DIR = Path("coco_subset_yolo")
SAFEWALK_DIR = Path("SafeWalkBD-9")
OUT_DIR = Path("merged_dataset")
VAL_RATIO = 0.15
random.seed(42)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def find_yaml(folder):
    for name in ["data.yaml", "dataset.yaml"]:
        p = folder / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No data.yaml/dataset.yaml found in {folder}")

def to_list(names):
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)

coco_yaml = load_yaml(find_yaml(COCO_DIR))
safewalk_yaml = load_yaml(find_yaml(SAFEWALK_DIR))
coco_classes = to_list(coco_yaml["names"])
safewalk_classes = to_list(safewalk_yaml["names"])

unified, seen = [], set()
for cls in coco_classes + safewalk_classes:
    key = cls.strip().lower()
    if key not in seen:
        seen.add(key); unified.append(cls)

def class_index_map(source_classes, unified_classes):
    idx_map = {}
    for i, cls in enumerate(source_classes):
        for j, u in enumerate(unified_classes):
            if u.strip().lower() == cls.strip().lower():
                idx_map[i] = j; break
    return idx_map

coco_map = class_index_map(coco_classes, unified)
safewalk_map = class_index_map(safewalk_classes, unified)

def collect_all_pairs(root):
    pairs = []
    exts = ["*.jpg", "*.jpeg", "*.png"]
    for ext in exts:
        for img_file in root.rglob(ext):
            parts = list(img_file.parts)
            if "images" not in parts:
                continue
            idx = len(parts) - 1 - parts[::-1].index("images")  # last occurrence of "images"
            lbl_parts = parts[:]
            lbl_parts[idx] = "labels"
            lbl_path = Path(*lbl_parts).with_suffix(".txt")
            pairs.append((img_file, lbl_path if lbl_path.exists() else None))
    return pairs

def remap_lines(lbl_file, idx_map):
    if lbl_file is None:
        return []
    lines_out = []
    for line in open(lbl_file):
        parts = line.strip().split()
        if not parts:
            continue
        new_cls = idx_map.get(int(parts[0]))
        if new_cls is not None:
            parts[0] = str(new_cls)
            lines_out.append(" ".join(parts))
    return lines_out

all_items = []
for img_file, lbl_file in collect_all_pairs(COCO_DIR):
    all_items.append((img_file, lbl_file, coco_map, "coco"))
for img_file, lbl_file in collect_all_pairs(SAFEWALK_DIR):
    all_items.append((img_file, lbl_file, safewalk_map, "sw"))

random.shuffle(all_items)
split_point = int(len(all_items) * (1 - VAL_RATIO))
train_items = all_items[:split_point]
val_items = all_items[split_point:]

def write_split(items, split_name):
    out_img = OUT_DIR / split_name / "images"
    out_lbl = OUT_DIR / split_name / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for img_file, lbl_file, idx_map, tag in items:
        new_name = f"{tag}_{img_file.name}"
        shutil.copy(img_file, out_img / new_name)
        lines = remap_lines(lbl_file, idx_map)
        (out_lbl / (Path(new_name).stem + ".txt")).write_text("\n".join(lines))

write_split(train_items, "train")
write_split(val_items, "val")

data_yaml = {"train": "train/images", "val": "val/images", "nc": len(unified), "names": unified}
with open(OUT_DIR / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"Total images pooled: {len(all_items)}")
print(f"Train: {len(train_items)} | Val: {len(val_items)}")
print(f"Unified classes ({len(unified)}): {unified}")
print("\nDone. Merged dataset at:", OUT_DIR.resolve())