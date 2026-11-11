from pathlib import Path
from statistics import median
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
CORPUS = Path("corpus")

for folder, pattern in [("product-docs", "*.md"), ("tickets", "*.html")]:
    files = sorted((CORPUS / folder).rglob(pattern))
    sizes = [len(enc.encode(f.read_text(encoding="utf-8", errors="ignore"))) for f in files]
    print(f"{folder:13} | {len(files):3} files | tokens min {min(sizes):5} "
          f"median {int(median(sizes)):5} max {max(sizes):5} total {sum(sizes):,}")

pdfs = list((CORPUS / "runbooks").rglob("*.pdf"))
print(f"runbooks      | {len(pdfs):3} PDFs (we'll measure these on Day 4)")