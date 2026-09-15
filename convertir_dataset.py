from pathlib import Path

LABEL_DIRS = [
    Path("dataset/train/labels"),
    Path("dataset/valid/labels"),
    Path("dataset/test/labels"),
]

converted = 0
converted_files = set()

for label_dir in LABEL_DIRS:
    if not label_dir.exists():
        continue

    for file in label_dir.glob("*.txt"):
        lines = file.read_text().splitlines()
        new_lines = []
        modified = False

        for line_number, line in enumerate(lines, start=1):
            values = line.split()

            # Una caja YOLO tiene exactamente 5 valores:
            # clase x_centro y_centro ancho alto
            if len(values) == 5:
                new_lines.append(line)
                continue

            # Una segmentación tiene:
            # clase x1 y1 x2 y2 x3 y3 ...
            try:
                cls = int(values[0])
                coords = list(map(float, values[1:]))
            except (ValueError, IndexError):
                new_lines.append(line)
                continue

            # Debe haber pares X,Y
            if len(coords) < 6 or len(coords) % 2 != 0:
                new_lines.append(line)
                continue

            xs = coords[0::2]
            ys = coords[1::2]

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            new_line = (
                f"{cls} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            new_lines.append(new_line)
            converted += 1
            modified = True

            print(
                f"Convertida: {file} "
                f"(línea {line_number})"
            )

        if modified:
            file.write_text("\n".join(new_lines) + "\n")
            converted_files.add(file)

print("\n=== RESULTADO ===")
print(f"Anotaciones convertidas: {converted}")
print(f"Archivos modificados:    {len(converted_files)}")