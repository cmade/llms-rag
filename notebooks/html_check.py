from pathlib import Path
from bs4 import BeautifulSoup
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
p = sorted(Path("corpus/tickets").rglob("*.html"))[0]
raw = p.read_text(encoding="utf-8")

soup = BeautifulSoup(raw, "html.parser")
print("tags used:", sorted({t.name for t in soup.find_all(True)}))
print("classes used:", sorted({c for t in soup.find_all(class_=True) for c in t["class"]}))

for tag in soup(["script", "style"]):
    tag.decompose()
text = soup.get_text(separator="\n", strip=True)

print(p.as_posix())
print("raw tokens:", len(enc.encode(raw)), "| cleaned tokens:", len(enc.encode(text)))
print("-" * 60)
print(text[:1500])