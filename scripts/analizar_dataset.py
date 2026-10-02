from pathlib import Path
from collections import Counter

LABEL_DIRS = [
    Path("dataset/train/labels"),
    Path("dataset/valid/labels"),
    Path("dataset/test/labels"),
]

names = [
    "Bacterial",
    "Downy_mildew",
    "Lettuce - Anthracnose",
    "Powdery_mildew",
    "Septoria_Blight",
    "healthy",
    "lettuce mosaic virus",
]

class_boxes = Counter()
class_files = Counter()

invalid = []
segments = []
empty = []

for label_dir in LABEL_DIRS:
    if not label_dir.exists():
        continue

    for file in label_dir.glob("*.txt"):
        lines = file.read_text().splitlines()

        if not lines:
            empty.append(file)
            continue

        seen_classes = set()

        for line_number, line in enumerate(lines, start=1):
            values = line.split()

            if len(values) != 5:
                segments.append((file, line_number, len(values), line))
                continue

            try:
                cls = int(values[0])
                coords = list(map(float, values[1:]))
            except ValueError:
                invalid.append((file, line_number, line))
                continue

            if cls < 0 or cls >= len(names):
                invalid.append((file, line_number, line))
                continue

            if not all(0 <= x <= 1 for x in coords):
                invalid.append((file, line_number, line))
                continue

            class_boxes[cls] += 1
            seen_classes.add(cls)

        for cls in seen_classes:
            class_files[cls] += 1


print("\n=== DISTRIBUCIÓN DE CLASES ===\n")

for cls, name in enumerate(names):
    print(
        f"{cls}: {name:25} "
        f"cajas={class_boxes[cls]:6} "
        f"imágenes={class_files[cls]:5}"
    )

print("\n=== PROBLEMAS ===\n")
print(f"Anotaciones con formato diferente a 5 valores: {len(segments)}")
print(f"Anotaciones inválidas:                    {len(invalid)}")
print(f"Archivos vacíos:                          {len(empty)}")

if segments:
    print("\nPrimeras anotaciones no estándar:")
    for item in segments[:10]:
        print(item)

if invalid:
    print("\nPrimeras anotaciones inválidas:")
    for item in invalid[:10]:
        print(item)