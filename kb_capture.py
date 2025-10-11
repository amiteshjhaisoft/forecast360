# kb_capture.py — capture-only KB writer (tables/figs/images/html/uploads) + TEXT SIDECARS + META INDEX
from __future__ import annotations

import io, re, json, hashlib, datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _now() -> str:
    # Use underscore in timestamp to avoid hyphens in filenames
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _txt(path: Path) -> Path:
    return path.with_suffix(".txt")


# ---------- capture model ----------
@dataclass
class _Item:
    # kinds: table | fig_png | image | html | upload | markdown  (note: fig_png = PNG bytes)
    kind: str
    title: str  # block name (used in filename)
    payload: Any
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_now)


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

    **New in this custom build**
    - Writes **sidecar text** for figs/images/html so embedders have semantics.
    - Writes a **blocks.md** concatenation of captured markdown.
    - Emits **meta/index.json** with one entry per artifact: kind, block, title, path, sidecar, created_at, extras.
    """

    def __init__(self, folder_name: str):
        if not folder_name or not str(folder_name).strip():
            raise ValueError("folder_name must be a non-empty string")
        self.folder_name = folder_name.strip()
        self._items: List[_Item] = []
        self._orig: Dict[str, Any] = {}
        self._patched = False

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
                r'<div[^>]*class="[^"]*block-card[^"]*"[^>]*>.*?<h[1-6][^>]*>(.*?)</h[1-6]> ',
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
                    if isinstance(obj, Axes):
                        obj = obj.figure
                    if isinstance(obj, Figure):
                        png = _fig_to_png_bytes(obj)
                        self._items.append(_Item("fig_png", _GLOBAL_BLOCK, png))
                        return True
                except Exception:
                    pass
            # images
            if _is_pil(obj) or _is_np(obj) or isinstance(obj, (bytes, bytearray)):
                self._items.append(_Item("image", _GLOBAL_BLOCK, obj))
                return True
            # plotly / altair / HTML-ish
            try:
                if hasattr(obj, "to_html") or hasattr(obj, "write_html"):
                    self._items.append(_Item("html", _GLOBAL_BLOCK, obj, meta={"format": "plotly_or_html"}))
                    return True
            except Exception:
                pass
            return False

        def _w_write(*args, **kwargs):
            for a in args:
                _capture_from_obj(a)
            return self._orig["write"](*args, **kwargs)

        # explicit Streamlit APIs
        def _w_dataframe(df, *a, **k):
            self._items.append(_Item("table", _GLOBAL_BLOCK, df))
            return self._orig["dataframe"](df, *a, **k)

        def _w_table(df, *a, **k):
            self._items.append(_Item("table", _GLOBAL_BLOCK, df))
            return self._orig["table"](df, *a, **k)

        def _w_pyplot(fig=None, *a, **k):
            # accept Figure or Axes or None; snapshot bytes NOW
            if fig is None and _plt is not None:
                fig = _plt.gcf()
            else:
                try:
                    from matplotlib.axes import Axes
                    if isinstance(fig, Axes):
                        fig = fig.figure
                except Exception:
                    pass
            if fig is not None:
                try:
                    png = _fig_to_png_bytes(fig)
                    self._items.append(_Item("fig_png", _GLOBAL_BLOCK, png))
                except Exception:
                    pass
            return self._orig["pyplot"](fig, *a, **k)

        def _w_image(img, *a, **k):
            payload = img
            if isinstance(payload, (list, tuple)):
                for i, x in enumerate(payload):
                    self._items.append(_Item("image", f"{_GLOBAL_BLOCK}-{i}", x))
            else:
                self._items.append(_Item("image", _GLOBAL_BLOCK, payload))
            return self._orig["image"](img, *a, **k)

        def _w_plotly(fig, *a, **k):
            self._items.append(_Item("html", _GLOBAL_BLOCK, fig, meta={"format": "plotly"}))
            return self._orig["plotly_chart"](fig, *a, **k)

        def _w_altair(chart, *a, **k):
            self._items.append(_Item("html", _GLOBAL_BLOCK, chart, meta={"format": "altair"}))
            return self._orig["altair_chart"](chart, *a, **k)

        def _w_uploader(label, *a, **k):
            res = self._orig["file_uploader"](label, *a, **k)
            if res is None:
                return res
            if isinstance(res, list):
                for up in res:
                    self._capture_upload(up)
            else:
                self._capture_upload(res)
            return res

        # attach wrappers
        if self._orig.get("title"):
            st.title = _w_title  # type: ignore
        if self._orig.get("header"):
            st.header = _w_header  # type: ignore
        if self._orig.get("subheader"):
            st.subheader = _w_subheader  # type: ignore
        if self._orig.get("markdown"):
            st.markdown = _w_markdown  # type: ignore
        if self._orig.get("write"):
            st.write = _w_write  # type: ignore
        if self._orig.get("dataframe"):
            st.dataframe = _w_dataframe  # type: ignore
        if self._orig.get("table"):
            st.table = _w_table  # type: ignore
        if self._orig.get("pyplot"):
            st.pyplot = _w_pyplot  # type: ignore
        if self._orig.get("image"):
            st.image = _w_image  # type: ignore
        if self._orig.get("plotly_chart"):
            st.plotly_chart = _w_plotly  # type: ignore
        if self._orig.get("altair_chart"):
            st.altair_chart = _w_altair  # type: ignore
        if self._orig.get("file_uploader"):
            st.file_uploader = _w_uploader  # type: ignore

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

        for it in self._items:
            block_slug = _slugify(it.title or "home")
            ts = it.ts

            if it.kind == "markdown":
                # Keep the raw markdown text in order (for blocks.md)
                body = str(it.payload)
                blocks_markdown.append(body if body.endswith("\n") else body + "\n")

            elif it.kind == "table":
                try:
                    name = f"{ts}_{block_slug}.csv"
                    csv_path = tables_dir / name
                    csv_path.write_text(it.payload.to_csv(index=False), encoding="utf-8")

                    # Sidecar Markdown describing the table
                    try:
                        cols = list(getattr(it.payload, "columns", []))
                        nrows = int(getattr(it.payload, "shape", [0])[0] or 0)
                    except Exception:
                        cols, nrows = [], 0
                    sc_path = _txt(csv_path.with_suffix(""))  # tables/<ts>_<block>.txt
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
                        "created_at": ts,
                    })
                except Exception:
                    pass

            elif it.kind == "fig_png":
                try:
                    png_path = figs_dir / f"{ts}_{block_slug}.png"
                    png_path.write_bytes(it.payload)

                    sc_path = _txt(png_path.with_suffix(""))
                    sc_text = (
                        f"# {it.title}\n\n"
                        f"Figure image exported from Forecast360 block **{it.title}**.\n"
                    )
                    sc_path.write_text(sc_text, encoding="utf-8")

                    index_items.append({
                        "kind": "fig",
                        "block": it.title,
                        "title": it.title,
                        "path": str(png_path.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "created_at": ts,
                    })
                except Exception:
                    pass

            elif it.kind == "image":
                try:
                    name = f"{ts}_{block_slug}.png"
                    img_path = images_dir / name
                    payload = it.payload
                    if _is_pil(payload):
                        buf = io.BytesIO()
                        payload.save(buf, format="PNG")
                        img_path.write_bytes(buf.getvalue())
                    elif _is_np(payload):
                        from PIL import Image
                        arr = payload
                        if getattr(arr, "dtype", None) != "uint8":
                            import numpy as np
                            arr = arr - arr.min()
                            rng = arr.max() - arr.min() or 1.0
                            arr = (255.0 * (arr / rng)).astype("uint8")
                        Image.fromarray(arr).save(img_path)
                    elif isinstance(payload, (bytes, bytearray)):
                        img_path.write_bytes(bytes(payload))

                    sc_path = _txt(img_path.with_suffix(""))
                    sc_text = f"# {it.title}\n\nImage associated with block **{it.title}**.\n"
                    sc_path.write_text(sc_text, encoding="utf-8")

                    index_items.append({
                        "kind": "image",
                        "block": it.title,
                        "title": it.title,
                        "path": str(img_path.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "created_at": ts,
                    })
                except Exception:
                    pass

            elif it.kind == "html":
                try:
                    name = f"{ts}_{block_slug}.html"
                    html_path = html_dir / name
                    src = it.payload
                    fmt = (it.meta or {}).get("format")

                    html_written = False
                    if fmt == "plotly":
                        if hasattr(src, "write_html"):
                            src.write_html(str(html_path), include_plotlyjs="cdn", full_html=False)
                            html_written = True
                        elif hasattr(src, "to_html"):
                            html_path.write_text(src.to_html(include_plotlyjs="cdn", full_html=False), encoding="utf-8")
                            html_written = True
                    elif fmt == "altair":
                        if hasattr(src, "to_html"):
                            html_path.write_text(src.to_html(), encoding="utf-8")
                            html_written = True

                    if not html_written:
                        # best-effort generic serialization
                        html_path.write_text(str(src), encoding="utf-8")

                    # Sidecar: text-only extraction to feed embedders
                    sc_path = _txt(html_path.with_suffix(""))
                    raw = ""
                    try:
                        if hasattr(src, "to_html"):
                            raw = src.to_html()
                        elif hasattr(src, "to_json"):
                            raw = src.to_json()
                        else:
                            raw = str(src)
                    except Exception:
                        raw = str(src)
                    plain = _clean_html_text(raw) or f"Interactive chart for block {it.title}."
                    sc_path.write_text(plain, encoding="utf-8")

                    index_items.append({
                        "kind": "html",
                        "block": it.title,
                        "title": it.title,
                        "path": str(html_path.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "created_at": ts,
                    })
                except Exception:
                    pass

            elif it.kind == "upload":
                try:
                    orig = it.meta.get("original") or f"{ts}_{block_slug}.bin"
                    # Replace hyphens with underscores in upload filenames
                    orig = orig.replace("-", "_")
                    target = upl_dir / orig
                    if target.exists():
                        stem = Path(orig).stem
                        suf = Path(orig).suffix
                        h = hashlib.sha256(it.payload).hexdigest()[:8]
                        target = upl_dir / f"{stem}__dup_{h}{suf}"
                    target.write_bytes(it.payload)

                    # Also create a tiny sidecar with filename and basic file facts
                    sc_path = _txt(target.with_suffix(""))
                    sc_text = (
                        f"# Upload: {orig}\n\n"
                        f"Uploaded in block **{it.title}**. Size: {len(it.payload)} bytes.\n"
                    )
                    sc_path.write_text(sc_text, encoding="utf-8")

                    index_items.append({
                        "kind": "upload",
                        "block": it.title,
                        "title": it.title,
                        "path": str(target.relative_to(outdir)),
                        "sidecar": str(sc_path.relative_to(outdir)),
                        "size": len(it.payload),
                        "created_at": ts,
                    })
                except Exception:
                    pass

        # ---- write composite text artifacts ----
        # 1) Blocks markdown (ordered capture of markdown bodies)
        if blocks_markdown:
            (text_dir / "blocks.md").write_text("".join(blocks_markdown), encoding="utf-8")

        # 2) Meta index (one JSON list)
        (meta_dir / "index.json").write_text(
            json.dumps(index_items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # clear buffer after flush
        self._items.clear()
        return str(outdir)
