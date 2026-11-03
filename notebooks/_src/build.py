"""Convert simple cell-delimited source files (src/*.txt) into .ipynb notebooks.

Format: a line that is exactly '%%md' starts a markdown cell, '%%code' starts a code cell.
Usage: python build.py <src_dir> <out_dir> [--execute]
"""
import sys, pathlib, nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def parse(text):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        if line.strip() in ("%%md", "%%code"):
            if kind and "".join(buf).strip():
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = line.strip()[2:], []
        else:
            buf.append(line)
    if kind and "".join(buf).strip():
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def build(src, out, execute):
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, body in parse(src.read_text()):
        nb.cells.append(new_markdown_cell(body) if kind == "md" else new_code_cell(body))
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(out.parent)}}).execute()
    nbformat.write(nb, out)
    print("built", out.name, len(nb.cells), "cells")


if __name__ == "__main__":
    src_dir, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    only = [a for a in sys.argv[3:] if not a.startswith("--")]
    for f in sorted(src_dir.glob("*.txt")):
        if only and not any(o in f.name for o in only):
            continue
        build(f, out_dir / (f.stem + ".ipynb"), "--execute" in sys.argv)
