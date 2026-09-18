"""Extract the canonical YAML code blocks (spec §23.1, §23.2) verbatim from the frozen spec DOCX."""
import sys, zipfile, xml.etree.ElementTree as ET
sys.path.insert(0, __import__("os").path.dirname(__file__))
from docx_extract import W, ptext, style
BLOCKS = {"model:": "configs/frozen/downstream_resnet18.yaml", "method: GPAT-B0": "configs/methods/gpat_b0.yaml"}
spec = sys.argv[1]
body = ET.fromstring(zipfile.ZipFile(spec).read("word/document.xml")).find(W + "body")
found = {}
for p in body.iter(W + "p"):
    if style(p) != "CodeBlock": continue
    t = ptext(p)
    for prefix, out in BLOCKS.items():
        if t.startswith(prefix):
            assert out not in found, f"duplicate block for {out}"
            found[out] = t
for out, t in found.items():
    with open(out, "w", encoding="utf-8") as f: f.write(t.rstrip("\n") + "\n")
    print("wrote", out)
assert len(found) == len(BLOCKS), found.keys()
