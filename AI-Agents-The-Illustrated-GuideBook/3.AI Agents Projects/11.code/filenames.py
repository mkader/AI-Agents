import re
from pathlib import Path


def documentation_path(docs_dir: Path, title: str) -> Path:
    """Build a unique Windows-safe MDX path without changing the document title."""
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", title.lower())
    stem = stem.strip(" ._")[:120].rstrip(" .")
    if not stem:
        raise ValueError("Document title contains no usable filename characters")

    if re.fullmatch(
        r"(con|prn|aux|nul|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])",
        stem.split(".")[0],
    ):
        stem = "_" + stem

    path = docs_dir / f"{stem}.mdx"
    index = 2
    while path.exists():
        path = docs_dir / f"{stem}_{index}.mdx"
        index += 1
    return path
