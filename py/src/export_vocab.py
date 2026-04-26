import json

# =========================
# // INIT
# =========================

with open("./data/lstm_export.json") as f:
    data = json.load(f)

vocab = data["vocab"]

lines = ["--!strict\n"]
lines.append("return {")
for word in vocab:
    escaped = word.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'\t"{escaped}",')
lines.append("}")

with open("./data/vocab.lua", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))