from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from roadsign_assist.datasets.split import build_grouped_splits
from roadsign_assist.paths import project_path


@dataclass(frozen=True)
class SourceBox:
    filename: str
    class_id: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def validate(self, width: int, height: int) -> None:
        if self.xmin < 0 or self.ymin < 0:
            raise ValueError(f"Negative EMTD box coordinate in {self.filename}")
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError(f"Non-positive EMTD box in {self.filename}")
        if self.xmax > width or self.ymax > height:
            raise ValueError(
                f"EMTD box exceeds image in {self.filename}: "
                f"{self.xmax}x{self.ymax} > {width}x{height}"
            )


@dataclass(frozen=True)
class EMTDClassMapping:
    semantic_sign_id: str | None
    parameter_value: float | None
    confidence: float


def read_source_boxes(path: str | Path) -> dict[str, list[SourceBox]]:
    resolved = project_path(path)
    grouped: dict[str, list[SourceBox]] = defaultdict(list)
    with resolved.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            box = SourceBox(
                filename=row["filename"].strip(),
                class_id=int(row["Class ID"]),
                xmin=int(row["xmin"]),
                ymin=int(row["ymin"]),
                xmax=int(row["xmax"]),
                ymax=int(row["ymax"]),
            )
            grouped[box.filename.casefold()].append(box)
    return grouped


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = gray.tobytes()
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def _duplicate_groups(hashes: dict[str, int], threshold: int) -> dict[str, str]:
    parents = {name: name for name in hashes}

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[max(root_left, root_right)] = min(root_left, root_right)

    names = sorted(hashes)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if (hashes[left] ^ hashes[right]).bit_count() <= threshold:
                union(left, right)
    return {name: f"emtd_dup_{find(name)}" for name in names}


def _copy_processed_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty EMTD output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_emtd_subset(
    *,
    subset_manifest_path: str | Path = "data/raw/emtd/metadata/subset_manifest.csv",
    ground_truth_path: str | Path = "data/raw/emtd/metadata/GT.csv",
    class_mapping_path: str | Path = "configs/catalogue/emtd_class_mapping.v0.1.json",
    dataset_manifest_path: str | Path = "data/manifests/dataset.csv",
    annotation_path: str | Path = "data/annotations/emtd_boxes.csv",
    split_root: str | Path = "data/splits",
    processed_root: str | Path = "data/processed",
    duplicate_threshold: int = 6,
    seed: int = 2513,
) -> dict[str, int]:
    subset_manifest = project_path(subset_manifest_path)
    source_boxes = read_source_boxes(ground_truth_path)
    class_mapping_payload = json.loads(project_path(class_mapping_path).read_text(encoding="utf-8"))
    class_mapping: dict[int, EMTDClassMapping] = {
        int(class_id): EMTDClassMapping(
            semantic_sign_id=(
                str(value["semantic_sign_id"]) if value["semantic_sign_id"] else None
            ),
            parameter_value=(
                float(value["parameter_value"]) if value["parameter_value"] is not None else None
            ),
            confidence=float(value["confidence"]),
        )
        for class_id, value in class_mapping_payload["classes"].items()
    }
    source_class_ids = {box.class_id for boxes in source_boxes.values() for box in boxes}
    if source_class_ids != set(class_mapping):
        raise ValueError(
            "EMTD class mapping must cover every source class exactly: "
            f"source_only={sorted(source_class_ids - set(class_mapping))}, "
            f"mapping_only={sorted(set(class_mapping) - source_class_ids)}"
        )
    with subset_manifest.open(newline="", encoding="utf-8") as handle:
        downloaded = list(csv.DictReader(handle))
    if not downloaded:
        raise ValueError("EMTD subset manifest is empty")

    image_records: list[dict[str, object]] = []
    box_records: list[dict[str, object]] = []
    hashes: dict[str, int] = {}
    image_paths: dict[str, Path] = {}

    for row in downloaded:
        image_path = project_path(row["relative_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"EMTD checksum mismatch: {image_path}")
        filename_key = row["filename"].casefold()
        boxes = source_boxes.get(filename_key, [])
        if not boxes:
            raise ValueError(f"No EMTD annotations for {row['filename']}")
        with Image.open(image_path) as image:
            width, height = image.size
        for box in boxes:
            box.validate(width, height)

        sample_id = f"emtd_{hashlib.sha1(filename_key.encode()).hexdigest()[:16]}"
        hashes[sample_id] = difference_hash(image_path)
        image_paths[sample_id] = image_path
        class_ids = sorted({box.class_id for box in boxes})
        mapped_semantic_values: set[str] = set()
        for value in class_ids:
            semantic_sign_id = class_mapping[value].semantic_sign_id
            if semantic_sign_id is not None:
                mapped_semantic_values.add(semantic_sign_id)
        mapped_semantics = sorted(mapped_semantic_values)
        image_records.append(
            {
                "sample_id": sample_id,
                "source_id": "emtd_zenodo_1217105",
                "path": image_path.as_posix(),
                "semantic_sign_id": " ".join(mapped_semantics),
                "group_id": "",
                "route_id": "",
                "session_id": "",
                "physical_sign_id": "",
                "has_sign": "true",
                "is_synthetic": "false",
                "annotation_status": "source_boxes_unreviewed",
            }
        )
        for instance_index, box in enumerate(boxes, start=1):
            mapped_class = class_mapping[box.class_id]
            box_records.append(
                {
                    "sample_id": sample_id,
                    "instance_id": f"{sample_id}_{instance_index:03d}",
                    "filename": row["filename"],
                    "class_id": box.class_id,
                    "provisional_class": f"emtd_class_{box.class_id:02d}",
                    "semantic_sign_id": mapped_class.semantic_sign_id or "",
                    "parameter_value": (
                        mapped_class.parameter_value
                        if mapped_class.parameter_value is not None
                        else ""
                    ),
                    "mapping_confidence": mapped_class.confidence,
                    "xmin": box.xmin,
                    "ymin": box.ymin,
                    "xmax": box.xmax,
                    "ymax": box.ymax,
                    "width": width,
                    "height": height,
                    "annotation_status": "source_box_unreviewed",
                }
            )

    groups = _duplicate_groups(hashes, duplicate_threshold)
    for record in image_records:
        record["group_id"] = groups[str(record["sample_id"])]

    dataset_manifest = project_path(dataset_manifest_path)
    annotations = project_path(annotation_path)
    _write_csv(dataset_manifest, image_records)
    _write_csv(annotations, box_records)
    split_rows = build_grouped_splits(
        dataset_manifest,
        split_root,
        seed=seed,
        train_fraction=0.70,
        validation_fraction=0.15,
    )

    split_by_sample = {
        row["sample_id"]: split for split, rows in split_rows.items() for row in rows
    }
    processed = project_path(processed_root)
    detection_root = processed / "emtd_detection"
    classification_root = processed / "emtd_classification"
    for target in (detection_root, classification_root):
        if target.exists():
            shutil.rmtree(target)

    boxes_by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in box_records:
        boxes_by_sample[str(row["sample_id"])].append(row)

    classification_count = 0
    classification_split_counts: Counter[str] = Counter()
    classification_label_counts: dict[str, Counter[str]] = {
        split: Counter() for split in split_rows
    }
    for sample_id, image_path in image_paths.items():
        split = split_by_sample[sample_id]
        suffix = image_path.suffix.lower() or ".jpg"
        destination_name = f"{sample_id}{suffix}"
        detection_image = detection_root / "images" / split / destination_name
        _copy_processed_image(image_path, detection_image)
        labels: list[str] = []
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            for instance in boxes_by_sample[sample_id]:
                xmin = int(str(instance["xmin"]))
                ymin = int(str(instance["ymin"]))
                xmax = int(str(instance["xmax"]))
                ymax = int(str(instance["ymax"]))
                center_x = ((xmin + xmax) / 2) / width
                center_y = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                labels.append(f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")

                margin_x = max(2, round((xmax - xmin) * 0.08))
                margin_y = max(2, round((ymax - ymin) * 0.08))
                crop = rgb.crop(
                    (
                        max(0, xmin - margin_x),
                        max(0, ymin - margin_y),
                        min(width, xmax + margin_x),
                        min(height, ymax + margin_y),
                    )
                )
                class_name = str(instance["semantic_sign_id"])
                if not class_name:
                    continue
                crop_path = (
                    classification_root / split / class_name / f"{instance['instance_id']}.jpg"
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(crop_path, quality=95)
                classification_count += 1
                classification_split_counts[split] += 1
                classification_label_counts[split][class_name] += 1
        label_path = detection_root / "labels" / split / f"{sample_id}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(labels) + "\n", encoding="ascii")

    data_yaml = {
        "path": detection_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": {0: "traffic_sign"},
    }
    (detection_root / "data.yaml").write_text(
        json.dumps(data_yaml, indent=2) + "\n",
        encoding="utf-8",
    )
    dataset_metadata = {
        "schema_version": "1.0",
        "source_id": "emtd_zenodo_1217105",
        "annotation_status": "source_boxes_unreviewed",
        "training_scope": "experimental_only",
        "images": len(image_records),
        "instances": len(box_records),
        "duplicate_groups": len(set(groups.values())),
        "coursework_images_included": 0,
        "class_mapping_status": class_mapping_payload["review_status"],
        "mapped_source_classes": sum(
            bool(value.semantic_sign_id) for value in class_mapping.values()
        ),
        "unmapped_source_classes": sum(
            not value.semantic_sign_id for value in class_mapping.values()
        ),
        "split_strategy": "deterministic_group_stratified",
        "split_images": {split: len(rows) for split, rows in split_rows.items()},
        "split_groups": {
            split: len({row["effective_group"] for row in rows})
            for split, rows in split_rows.items()
        },
        "subset_manifest_sha256": hashlib.sha256(subset_manifest.read_bytes()).hexdigest(),
        "class_mapping_sha256": hashlib.sha256(
            project_path(class_mapping_path).read_bytes()
        ).hexdigest(),
    }
    (detection_root / "dataset_metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    vocabulary = sorted(
        {str(row["semantic_sign_id"]) for row in box_records if row["semantic_sign_id"]}
    )
    (classification_root / "labels.json").write_text(
        json.dumps(vocabulary, indent=2) + "\n",
        encoding="utf-8",
    )
    (classification_root / "dataset_metadata.json").write_text(
        json.dumps(
            {
                **dataset_metadata,
                "classification_crops": classification_count,
                "classification_split_crops": dict(sorted(classification_split_counts.items())),
                "classification_split_label_counts": {
                    split: dict(sorted(counts.items()))
                    for split, counts in classification_label_counts.items()
                },
                "missing_train_labels": sorted(
                    set(vocabulary) - set(classification_label_counts["train"])
                ),
                "missing_validation_labels": sorted(
                    set(vocabulary) - set(classification_label_counts["validation"])
                ),
                "missing_test_labels": sorted(
                    set(vocabulary) - set(classification_label_counts["test"])
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "images": len(image_records),
        "instances": len(box_records),
        "duplicate_groups": len(set(groups.values())),
        "classification_crops": classification_count,
    }
