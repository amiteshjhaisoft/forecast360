# Author: Amitesh Jha | iSOFT
# kb_capture.py — capture-only KB writer (tables/figs/images/html/uploads) + TEXT SIDECARS + META INDEX + VERSION
# NOTE: This version writes files WITHOUT datetime stamps. It overwrites stable names
# like tables/<block>.csv, figs/<block>.png, images/<block>.png, html/<block>.html.

from __future__ import annotations

import io, re, json, hashlib, datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# ---- optional deps (module still importable without them) ----
try:
    import streamlit as st
except Exception:
    st = None  # type: ignore

try:
    import matplotlib.pyplot as _plt
except Exception:
    _plt = None

try:
    import altair as _alt  # noqa
except Exception:
    _alt = None

# ---------- utils ----------
_SLUG = re.compile(r"[^a-z0-9]+", re.I)


def _slugify(s: str, max_len: int = 64) -> str:
    s = (_SLUG.sub("_", str(s).strip().lower())).strip("_")
    return s[:max_len] or "untitled"


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_pil(x: Any) -> bool:
    try:
        from PIL import Image
        return isinstance(x, Image.Image)
    except Exception:
        return False


def _is_np(x: Any) -> bool:
    try:
        import numpy as np
        return isinstance(x, np.ndarray)
    except Exception:
        return False


def _clean_html_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fig_to_png_bytes(fig) -> bytes:
    """Render a Matplotlib Figure to PNG bytes RIGHT NOW (safe if Streamlit later clears it)."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        FigureCanvas(fig)  # attach Agg canvas if missing
    except Exception:
        pass
    try:
        fig.canvas.draw()
    except Exception:
        pass
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    return buf.getvalue()


def _to_png_bytes_from_image_like(x: Any) -> Optional[bytes]:
    try:
        from PIL import Image
        if _is_pil(x):
            b = io.BytesIO()
            x.save(b, format="PNG")
            return b.getvalue()
        if _is_np(x):
            im = Image.fromarray(x)
            b = io.BytesIO()
            im.save(b, format="PNG")
            return b.getvalue()
    except Exception:
        return None
    return None


def _txt(path: Path) -> Path:
    return path.with_suffix(".txt")


def kb_compute_version(kb_dir: Path) -> str:
    """
    Compute a short content version for the KB using blocks.md + index.json.
    Returns a 12-char hex prefix of sha256 to be used as a cache/version key.
    """
    kb_dir = Path(kb_dir)
    h = hashlib.sha256()
    p_blocks = kb_dir / "text" / "blocks.md"
    p_index = kb_dir / "meta" / "index.json"
    if p_blocks.exists():
        h.update(p_blocks.read_bytes())
    if p_index.exists():
        h.update(p_index.read_bytes())
    return h.hexdigest()[:12]


# ---------- capture model ----------
@dataclass
class _Item:
    # kinds: table | fig_png | image | html | upload | markdown  (note: fig_png = PNG bytes)
    kind: str
    title: str  # block name (used in filename base)
    payload: Any
    meta: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)


_GLOBAL_BLOCK = "home"


def kb_set_block(name: str) -> None:
    """Manually set the logical block name used in filenames."""
    global _GLOBAL_BLOCK
    _GLOBAL_BLOCK = name.strip() or "home"


# ---------- main ----------
class KBCapture:
    """
    Minimal, capture-only KB writer — **customized** to be RAG-friendly.

    folder_name: fixed KB folder (e.g., "KB").
    Creates these subfolders:
      tables/, figs/, images/, html/, uploads/, text/, meta/

    - Writes **sidecar text** for figs/images/html so embedders have semantics.
    - Writes a **text/blocks.md** concatenation of captured markdown.
    - Emits **meta/index.json** with one entry per artifact.
    - Emits **meta/version.json** with {"version": "<12hex>", "created_at": "..."}.

    IMPORTANT: Filenames have **NO DATETIME STAMPS**. A stable base name is used:
      tables/<block>.csv, figs/<block>.png, images/<block>.png, html/<block>.html
    When multiple artifacts of the same kind appear for one block **in a single flush**,
    numeric suffixes are added: <block>_2.csv, <block>_3.csv, ...
    Subsequent flushes will overwrite the same base names (or the highest suffix seen in that flush).
    """

    def __init__(self, folder_name: str):
        if not folder_name or not str(folder_name).strip():
            raise ValueError("folder_name must be a non-empty string")
        self.folder_name = folder_name.strip()
        self._items: List[_Item] = []
        self._orig: Dict[str, Any] = {}
        self._patched = False
        self.last_version: Optional[str] = None  # filled on flush()

    # ----- patch -----
    def patch(self) -> "KBCapture":
        if st is None:
            raise RuntimeError("Streamlit is not available.")
        if self._patched:
            return self

        # keep originals
        names = [
            "markdown",
            "write",
            "title",
            "header",
            "subheader",
            "dataframe",
            "table",
            "pyplot",
            "image",
            "plotly_chart",
            "altair_chart",
            "file_uploader",
        ]
        for n in names:
            self._orig[n] = getattr(st, n, None)

        # headings set the active block
        def _push_block(level_prefix: str, txt: Any):
            title = str(txt).strip()
            if title:
                self._items.append(_Item("markdown", title, f"{level_prefix} {title}"))
                kb_set_block(title)

        def _w_title(txt, *a, **k):
            _push_block("#", txt)
            return self._orig["title"](txt, *a, **k)

        def _w_header(txt, *a, **k):
            _push_block("##", txt)
            return self._orig["header"](txt, *a, **k)

        def _w_subheader(txt, *a, **k):
            _push_block("###", txt)
            return self._orig["subheader"](txt, *a, **k)

        # Extract title from markdown or HTML (supports your block-card+h4)
        def _extract_title_from_markdown(md: str) -> Optional[str]:
            s = str(md)
            # 1) Any HTML <h1>..</h6>
            m = re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", s, flags=re.I | re.DOTALL)
            if m:
                return _slugify(_clean_html_text(m[-1]))
            # 2) Specific: <div class="block-card">...<h#>Title</h#>...</div>
            m2 = re.findall(
                r'<div[^>]*class="[^"]*block-card[^"]*"[^>]*>.*?<h[1-6][^>]*>(.*?)</h[1-6]>',
                s,
                flags=re.I | re.DOTALL,
            )
            if m2:
                return _slugify(_clean_html_text(m2[-1]))
            # 3) Fallback: last ATX (#) heading in text
            for line in s.splitlines()[::-1]:
                t = line.strip()
                if t.startswith("#"):
                    return _slugify(t.lstrip("#").strip())
            return None

        def _w_markdown(body, *a, **k):
            title = _extract_title_from_markdown(body)
            if title:
                # store the original body as a markdown marker (not yet written to disk)
                self._items.append(_Item("markdown", title, str(body)))
                kb_set_block(title)
            return self._orig["markdown"](body, *a, **k)

        # st.write auto-capture (df / fig / image / plotly / altair)
        def _capture_from_obj(obj: Any) -> bool:
            # pandas DataFrame-like
            try:
                if hasattr(obj, "to_csv"):
                    self._items.append(_Item("table", _GLOBAL_BLOCK, obj))
                    return True
            except Exception:
                pass
            # matplotlib figure/axes -> capture PNG BYTES NOW
            if _plt is not None:
                try:
                    from matplotlib.figure import Figure
                    from matplotlib.axes import Axes
                    if isinstance(obj, Figure):
                        self._items.append(_Item("fig_png", _GLOBAL_BLOCK, _fig_to_png_bytes(obj)))
                        return True
                    if isinstance(obj, Axes):
                        self._items.append(_Item("fig_png", _GLOBAL_BLOCK, _fig_to_png_bytes(obj.figure)))
                        return True
                except Exception:
                    pass
            # PIL / numpy arrays -> image PNG
            b = _to_png_bytes_from_image_like(obj)
            if b is not None:
                self._items.append(_Item("image", _GLOBAL_BLOCK, b))
                return True
            return False

        def _w_write(obj: Any, *a, **k):
            try:
                if _capture_from_obj(obj):
                    return self._orig["write"](obj, *a, **k)
            except Exception:
                pass
            return self._orig["write"](obj, *a, **k)

        def _w_dataframe(df, *a, **k):
            self._items.append(_Item("table", _GLOBAL_BLOCK, df))
            return self._orig["dataframe"](df, *a, **k)

        def _w_table(df, *a, **k):
            self._items.append(_Item("table", _GLOBAL_BLOCK, df))
            return self._orig["table"](df, *a, **k)

        def _w_pyplot(fig=None, *a, **k):
            try:
                if fig is None and _plt is not None:
                    fig = _plt.gcf()
                if fig is not None:
                    self._items.append(_Item("fig_png", _GLOBAL_BLOCK, _fig_to_png_bytes(fig)))
            except Exception:
                pass
            return self._orig["pyplot"](fig, *a, **k)

        def _w_image(img, *a, **k):
            try:
                b = _to_png_bytes_from_image_like(img)
                if b is None and hasattr(img, "read"):
                    b = img.read()
                if b is not None:
                    self._items.append(_Item("image", _GLOBAL_BLOCK, b))
            except Exception:
                pass
            return self._orig["image"](img, *a, **k)

        def _w_plotly(fig, *a, **k):
            try:
                # Save Plotly as standalone HTML (no timestamp)
                html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                self._items.append(_Item("html", _GLOBAL_BLOCK, html, meta={"format": "plotly"}))
            except Exception:
                pass
            return self._orig["plotly_chart"](fig, *a, **k)

        def _w_altair(chart, *a, **k):
            try:
                html = chart.to_html()
                self._items.append(_Item("html", _GLOBAL_BLOCK, html, meta={"format": "altair"}))
            except Exception:
                pass
            return self._orig["altair_chart"](chart, *a, **k)

        def _w_uploader(*a, **k):
            uploaded = self._orig["file_uploader"](*a, **k)
            try:
                if uploaded is not None:
                    if not isinstance(uploaded, list):
                        uploaded = [uploaded]
                    for u in uploaded:
                        self._capture_upload(u)
            except Exception:
                pass
            return uploaded

        # patch
        if self._orig.get("markdown"): st.markdown = _w_markdown  # type: ignore
        if self._orig.get("write"): st.write = _w_write  # type: ignore
        if self._orig.get("title"): st.title = _w_title  # type: ignore
        if self._orig.get("header"): st.header = _w_header  # type: ignore
        if self._orig.get("subheader"): st.subheader = _w_subheader  # type: ignore
        if self._orig.get("dataframe"): st.dataframe = _w_dataframe  # type: ignore
        if self._orig.get("table"): st.table = _w_table  # type: ignore
        if self._orig.get("pyplot"): st.pyplot = _w_pyplot  # type: ignore
        if self._orig.get("image"): st.image = _w_image  # type: ignore
        if self._orig.get("plotly_chart"): st.plotly_chart = _w_plotly  # type: ignore
        if self._orig.get("altair_chart"): st.altair_chart = _w_altair  # type: ignore
        if self._orig.get("file_uploader"): st.file_uploader = _w_uploader  # type: ignore

        self._patched = True
        return self

    def unpatch(self) -> None:
        if not self._patched:
            return
        for n, fn in self._orig.items():
            if fn is not None:
                setattr(st, n, fn)
        self._patched = False

    # ----- helpers -----
    def _capture_upload(self, uploaded: Any) -> None:
        try:
            fname = getattr(uploaded, "name", None)
            data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else None
            if fname and isinstance(data, (bytes, bytearray)):
                self._items.append(_Item("upload", _GLOBAL_BLOCK, bytes(data), meta={"original": fname}))
        except Exception:
            pass

    # ----- write -----
    def _outdir(self) -> Path:
        return Path(self.folder_name)  # fixed folder (no timestamped parent)

    def _next_name(self, counters: Dict[Tuple[str, str], int], kind: str, block_slug: str, ext: str) -> str:
        """Stable base name with incremental suffix within a single flush.
        First occurrence => <block>.<ext>
        2nd => <block>_2.<ext>, then _3, ...
        """
        key = (kind, block_slug)
        counters[key] += 1
        idx = counters[key]
        base = f"{block_slug}"
        if idx == 1:
            return f"{base}.{ext}"
        return f"{base}_{idx}.{ext}"

    def flush(self) -> str:
        outdir = _ensure_dir(self._outdir())

        # subdirs
        tables_dir = _ensure_dir(outdir / "tables")
        figs_dir = _ensure_dir(outdir / "figs")
        images_dir = _ensure_dir(outdir / "images")
        html_dir = _ensure_dir(outdir / "html")
        upl_dir = _ensure_dir(outdir / "uploads")
        text_dir = _ensure_dir(outdir / "text")
        meta_dir = _ensure_dir(outdir / "meta")

        # accumulators
        blocks_markdown: List[str] = []  # for text/blocks.md
        index_items: List[Dict[str, Any]] = []

        # counters for duplicate names within a single flush
        counters: Dict[Tuple[str, str], int] = defaultdict(int)

        for it in self._items:
            block_slug = _slugify(it.title or "home")

            if it.kind == "markdown":
                # Keep the raw markdown text in order (for blocks.md)
                body = str(it.payload)
                blocks_markdown.append(body if body.endswith("\n") else body + "\n")
                continue

            if it.kind == "table":
                try:
                    name = self._next_name(counters, "table", block_slug, "csv")
                    csv_path = tables_dir / name
                    csv_path.write_text(it.payload.to_csv(index=False), encoding="utf-8")

                    # Sidecar Markdown describing the table
                    try:
                        cols = list(getattr(it.payload, "columns", []))
                        nrows = int(getattr(it.payload, "shape", [0])[0] or 0)
                    except Exception:
                        cols, nrows = [], 0
                    sc_path = _txt(csv_path.with_suffix(""))  # tables/<block>[_2].txt
                    sc_text = (
                        f"# {it.title}\n\n"
                        f"Table captured from block **{it.title}**.\n\n"
                        f"Rows: {nrows}\n\nColumns: {', '.join(map(str, cols)) if cols else '(unknown)'}\n"
                    )
                    sc_path.write_text(sc_text, encoding="utf-8")

                    index_items.append({
                        "kind": "table",
                        "block": it.title,
                        "title": it.title,
                        "path": str(csv_path.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "nrows": nrows,
                        "columns": cols,
                        "created_at": it.created_at,
                    })
                except Exception:
                    pass
                continue

            if it.kind == "fig_png":
                try:
                    png_path = figs_dir / self._next_name(counters, "fig", block_slug, "png")
                    png_path.write_bytes(it.payload)

                    sc_path = _txt(png_path.with_suffix(""))
                    sc_text = (
                        f"# {it.title}\n\n"
                        f"Figure captured from block **{it.title}**.\n"
                    )
                    sc_path.write_text(sc_text, encoding="utf-8")

                    index_items.append({
                        "kind": "figure",
                        "block": it.title,
                        "title": it.title,
                        "path": str(png_path.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "created_at": it.created_at,
                    })
                except Exception:
                    pass
                continue

            if it.kind == "image":
                try:
                    img_path = images_dir / self._next_name(counters, "image", block_slug, "png")
                    img_path.write_bytes(it.payload)

                    sc_path = _txt(img_path.with_suffix(""))
                    sc_text = (
                        f"# {it.title}\n\n"
                        f"Image captured from block **{it.title}**.\n"
                    )
                    sc_path.write_text(sc_text, encoding="utf-8")

                    index_items.append({
                        "kind": "image",
                        "block": it.title,
                        "title": it.title,
                        "path": str(img_path.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "created_at": it.created_at,
                    })
                except Exception:
                    pass
                continue

            if it.kind == "html":
                try:
                    html_path = html_dir / self._next_name(counters, "html", block_slug, "html")
                    html_path.write_text(str(it.payload), encoding="utf-8")

                    sc_path = _txt(html_path.with_suffix(""))
                    sc_text = (
                        f"# {it.title}\n\n"
                        f"HTML artifact captured from block **{it.title}**.\n"
                    )
                    sc_path.write_text(sc_text, encoding="utf-8")

                    index_items.append({
                        "kind": "html",
                        "block": it.title,
                        "title": it.title,
                        "path": str(html_path.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "created_at": it.created_at,
                        "format": it.meta.get("format"),
                    })
                except Exception:
                    pass
                continue

            if it.kind == "upload":
                try:
                    original = _slugify(it.meta.get("original", "upload"), 80)
                    base = original.rsplit(".", 1)
                    if len(base) == 2:
                        bare, ext = base[0], base[1]
                    else:
                        bare, ext = original, "bin"
                    # Use original name, but avoid clobber within same flush
                    key = ("upload", bare)
                    counters[key] += 1
                    idx = counters[key]
                    if idx == 1:
                        name = f"{bare}.{ext}"
                    else:
                        name = f"{bare}_{idx}.{ext}"
                    upath = upl_dir / name
                    upath.write_bytes(it.payload)

                    sc_path = _txt(upath.with_suffix(""))
                    sc_text = (
                        f"# {it.title}\n\n"
                        f"Uploaded file from block **{it.title}**. Original: {it.meta.get('original')}\n"
                    )
                    sc_path.write_text(sc_text, encoding="utf-8")

                    index_items.append({
                        "kind": "upload",
                        "block": it.title,
                        "title": it.title,
                        "path": str(upath.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "created_at": it.created_at,
                        "original": it.meta.get("original"),
                    })
                except Exception:
                    pass
                continue

        # Write text/blocks.md
        (text_dir / "blocks.md").write_text("".join(blocks_markdown), encoding="utf-8")

        # Write meta/index.json
        (meta_dir / "index.json").write_text(json.dumps(index_items, indent=2), encoding="utf-8")

        # Compute and write version
        version = kb_compute_version(outdir)
        self.last_version = version
        (meta_dir / "version.json").write_text(
            json.dumps({"version": version, "created_at": _now_iso()}, indent=2),
            encoding="utf-8",
        )

        # Clear items for next cycle (optional — comment if you prefer accumulation)
        self._items.clear()

        return str(outdir.resolve())
