from pathlib import Path

for p in sorted(Path("corpus/product-docs").rglob("*.md"))[:6]:
    heads = [line for line in p.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    print(p.as_posix(), "→", heads)