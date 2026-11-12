"""Convert LabelMe JSON + images into a YOLO detection dataset (Python 3.9+)."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def convert_shapes(data):
    width, height = float(data['imageWidth']), float(data['imageHeight'])
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError('Invalid image dimensions')
    boxes = []
    clipped = 0
    for shape in data['shapes']:
        label = shape['label']
        if not isinstance(label, str) or not label.strip():
            raise ValueError('Empty or invalid class name')
        kind, points = shape.get('shape_type', 'polygon'), shape['points']
        if kind not in ('rectangle', 'polygon'):
            raise ValueError(f'Unsupported shape_type: {kind}')
        if (kind == 'rectangle' and len(points) != 2) or (kind == 'polygon' and len(points) < 3):
            raise ValueError(f'Invalid {kind} points')
        if any(len(p) != 2 for p in points):
            raise ValueError('Each point must contain x and y')
        xs, ys = zip(*[(float(p[0]), float(p[1])) for p in points])
        if not all(math.isfinite(v) for v in (*xs, *ys)):
            raise ValueError('Non-finite coordinates')
        raw = (min(xs), min(ys), max(xs), max(ys))
        x1, y1, x2, y2 = max(0, raw[0]), max(0, raw[1]), min(width, raw[2]), min(height, raw[3])
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Zero-area box or box outside image')
        clipped += raw != (x1, y1, x2, y2)
        boxes.append((label, ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)))
    return boxes, clipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=r"D:\Mystudy\yolo_orignal_data\RoadDataset",help="输入路径")
    parser.add_argument('--output', type=Path,default=r"D:\Mystudy\yolov5_datasets\RoadDataset", help='Default: INPUT/yolo_dataset; must not exist')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--classes', type=Path, help='UTF-8 class names, one per line; line order defines IDs')
    parser.add_argument('--skip-invalid', action='store_true',default=True, help='Skip invalid image/JSON pairs and report them')
    args = parser.parse_args()
    source = args.input.resolve()
    output = (args.output or source / 'yolo_dataset').resolve()
    if not source.is_dir() or not 0 < args.val_ratio < 1:
        parser.error('Input must be a directory; val-ratio must be between 0 and 1')
    if output.exists():
        parser.error(f'Output already exists; choose a new output directory: {output}')
    paths = sorted(source.rglob('*.json'))
    image_index = {}
    for image_path in source.rglob('*'):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            image_index.setdefault((image_path.parent, image_path.stem), []).append(image_path)
    records, errors, used_images = [], [], set()
    clipped = 0
    shape_types = Counter()
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict) or 'shapes' not in data:
                raise ValueError('Not a LabelMe annotation')
            candidates = image_index.get((path.parent, path.stem), [])
            if not candidates:
                # LabelMe may store an old absolute path; only use its basename locally.
                name = str(data.get('imagePath', '')).replace('\\', '/').split('/')[-1]
                candidate = path.parent / name
                candidates = [candidate] if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS else []
            if len(candidates) != 1:
                raise ValueError(f'Expected one matching image, found {len(candidates)}')
            image = candidates[0].resolve()
            if image in used_images:
                raise ValueError('Multiple JSON files reference the same image')
            boxes, count = convert_shapes(data)
            # Pillow is optional; when present, check actual image dimensions and integrity.
            try:
                from PIL import Image
            except ImportError:
                pass
            else:
                with Image.open(image) as im:
                    if im.size != (data['imageWidth'], data['imageHeight']):
                        raise ValueError(f'Image size {im.size} differs from JSON dimensions')
                    im.verify()
            relative = path.relative_to(source)
            name = path.stem if len(relative.parts) == 1 else path.stem + '_' + hashlib.sha256(relative.as_posix().encode()).hexdigest()[:16]
            records.append((image, name, boxes, relative.as_posix()))
            used_images.add(image)
            clipped += count
            shape_types.update(s.get('shape_type', 'polygon') for s in data['shapes'])
        except Exception as exc:
            errors.append({'json': str(path.relative_to(source)), 'error': str(exc)})
    if errors and not args.skip_invalid:
        print(json.dumps(errors[:20], ensure_ascii=False, indent=2))
        parser.error(f'{len(errors)} invalid annotations; fix them or explicitly use --skip-invalid')
    if len(records) < 2:
        parser.error('At least two valid image/JSON pairs are required')
    names = sorted({label for _, _, boxes, _ in records for label, _ in boxes})
    if args.classes:
        names = [line.strip() for line in args.classes.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not names or len(set(names)) != len(names):
        parser.error('Class list must be nonempty and contain no duplicates')
    mapping = {name: index for index, name in enumerate(names)}
    unknown = {label for _, _, boxes, _ in records for label, _ in boxes} - mapping.keys()
    if unknown:
        parser.error(f'Classes absent from class list: {unknown}')
    if len({name.casefold() for _, name, _, _ in records}) != len(records):
        parser.error('Output filename collision')
    random.Random(args.seed).shuffle(records)
    val_count = min(len(records)-1, max(1, round(len(records)*args.val_ratio)))
    counts, objects, manifest = Counter(), Counter(), []
    for split in ('train', 'valid'):
        for folder in ('images', 'labels'):
            (output / split / folder).mkdir(parents=True, exist_ok=False)
    for index, (image, name, boxes, annotation) in enumerate(records):
        split = 'valid' if index < val_count else 'train'
        shutil.copy2(image, output / split / 'images' / (name + image.suffix.lower()))
        lines = [str(mapping[label]) + ' ' + ' '.join(f'{v:.10f}' for v in box) for label, box in boxes]
        (output / split / 'labels' / (name + '.txt')).write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
        counts[split] += 1
        objects.update(label for label, _ in boxes)
        manifest.append({'json': annotation, 'split': split, 'name': name})
    yaml = [
        f"train: {(output / 'train' / 'images').as_posix()}",
        f"val: {(output / 'valid' / 'images').as_posix()}",
        f"nc: {len(names)}",
        "names: [" + ", ".join(
            "'" + name.replace("'", "''") + "'" for name in names
        ) + "]",
    ]
    (output / 'data.yaml').write_text('\n'.join(yaml)+'\n', encoding='utf-8')
    (output / 'classes.txt').write_text('\n'.join(names)+'\n', encoding='utf-8')
    unannotated = [str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS and output not in p.parents and p.resolve() not in used_images]
    report = dict(input=str(source), output=str(output), seed=args.seed, val_ratio=args.val_ratio, images=dict(counts), classes=mapping, objects=dict(objects), shape_types=dict(shape_types), clipped_boxes=clipped, empty_labels=sum(not r[2] for r in records), skipped=errors, images_without_valid_annotation=unannotated, manifest=manifest)
    (output / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('manifest', 'images_without_valid_annotation', 'skipped')}, ensure_ascii=False, indent=2))
    print(f'Skipped invalid JSON: {len(errors)}; images without valid annotation: {len(unannotated)}')


if __name__ == '__main__':
    main()
