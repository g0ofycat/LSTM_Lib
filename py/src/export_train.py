import json
import os

# =========================
# // CONSTANTS
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(SCRIPT_DIR, "../data/lstm_export.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../data/luau_export")
WEIGHT_COUNT = 9

# =========================
# // HELPERS
# =========================

def matrix_to_luau_inline(matrix: list[list[float]]) -> str:
    rows = ["{" + ",".join(f"{v:.6g}" for v in row) + "}" for row in matrix]
    return "{\n\t" + ",\n\t".join(rows) + "\n}"

def matrix_to_luau(name: str, matrix: list[list[float]]) -> str:
    rows = ["{" + ",".join(f"{v:.6g}" for v in row) + "}" for row in matrix]
    return f"local {name} = {{\n\t" + ",\n\t".join(rows) + "\n}"

def col_to_luau(col: list[list[float]]) -> str:
    return "{" + ",".join(f"{{{v[0]:.6g}}}" for v in col) + "}"

def zeros_to_luau(col: list) -> str:
    return "{" + ",".join("{0}" for _ in col) + "}"

# =========================
# // EXPORT
# =========================

def export_weights(data: dict, out_dir: str) -> None:
    for i in range(1, WEIGHT_COUNT + 1):
        matrix  = data["weights"][i - 1]
        name    = f"w{i}"
        content = f"{matrix_to_luau(name, matrix)}\n\nreturn {name}"
        path    = os.path.join(out_dir, f"{name}.lua")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"export_weights(): {name}.lua: {os.path.getsize(path) / 1024:.0f} KB")

def export_meta(data: dict, out_dir: str) -> None:
    lines = [
        "return {",
        f"\tinput_neurons = {data['input_neurons']},",
        f"\thidden_neurons = {data['hidden_neurons']},",
        f"\toutput_neurons = {data['output_neurons']},",
        f"\tlearning_rate = {data['learning_rate']},",
        f"\tdropout_rate = {data['dropout_rate']},",
        f"\tbf = {col_to_luau(data['bf'])},",
        f"\tbi = {col_to_luau(data['bi'])},",
        f"\tbg = {col_to_luau(data['bg'])},",
        f"\tbo = {col_to_luau(data['bo'])},",
        f"\tby = {col_to_luau(data['by'])},",
        f"\th = {zeros_to_luau(data['h'])},",
        f"\tc = {zeros_to_luau(data['c'])},",
        f"\tembed_weight = {matrix_to_luau_inline(data['embed'])},",
        "\tcache = {inputs={},h={},c={},f={},i={},g={},o={},y={},h_prev={},c_prev={}},",
        "}"
    ]

    path = os.path.join(out_dir, "meta.lua")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"export_meta(): meta.lua: {os.path.getsize(path) / 1024:.0f} KB")

# =========================
# // MAIN
# =========================

def main() -> None:
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    export_weights(data, OUTPUT_DIR)
    export_meta(data, OUTPUT_DIR)

if __name__ == "__main__":
    main()