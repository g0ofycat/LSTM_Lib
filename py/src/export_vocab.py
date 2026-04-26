import os
import json

# =========================
# // INIT
# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "../data/lstm_export.json")) as f:
    data = json.load(f)

vocab = data["vocab"]

lines = ["--!strict\n"]
lines.append("return {")
for word in vocab:
    escaped = word.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'\t"{escaped}",')
lines.append("}")

with open(os.path.join(SCRIPT_DIR, "../data/vocab.lua"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))