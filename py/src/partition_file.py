import os

# =========================
# // CONSTANTS
# =========================

INPUT_PATH    = "./data/luau_export/w9.lua"
OUTPUT_DIR    = "./data/parts"
LINES_PER_PART = 1000

# =========================
# // MAIN
# =========================

def main() -> None:
    with open(INPUT_PATH, encoding="utf-8") as f:
        lines = f.readlines()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stem, ext = os.path.splitext(os.path.basename(INPUT_PATH))
    total = (len(lines) + LINES_PER_PART - 1) // LINES_PER_PART

    for i in range(total):
        chunk = lines[i * LINES_PER_PART : (i + 1) * LINES_PER_PART]
        path  = os.path.join(OUTPUT_DIR, f"{stem}_part{i}{ext}")

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(chunk)

        print(f"main(): part {i}: {len(chunk)} lines -> {path}")

if __name__ == "__main__":
    main()