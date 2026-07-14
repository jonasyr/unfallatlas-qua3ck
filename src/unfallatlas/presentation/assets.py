from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import plotly
from nbformat import NotebookNode

from unfallatlas.presentation.models import AssetRecord, NotebookAnalysis

PACKAGE_DIR = Path(__file__).parent
STATIC_DIR = PACKAGE_DIR / "static"
VENDOR_DIR = PACKAGE_DIR / "vendor"
MATHJAX_FILENAME = "mathjax-3.2.2-tex-svg-full.js"
PLOTLY_MIME = "application/vnd.plotly.v1+json"


@dataclass(frozen=True, slots=True)
class PreparedNotebook:
    notebook: NotebookNode
    assets: tuple[AssetRecord, ...]


def write_atomic(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


class AssetStore:
    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)

    def put_bytes(
        self,
        *,
        namespace: str,
        stem: str,
        suffix: str,
        data: bytes,
        media_type: str,
        kind: str,
        cell_index: int | None,
    ) -> AssetRecord:
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("assets") / namespace / f"{stem}-{digest[:16]}{suffix}"
        write_atomic(self.output_root / relative, data)
        return AssetRecord(relative, digest, len(data), media_type, kind, cell_index)

    def put_named_bytes(
        self,
        *,
        relative_path: Path,
        data: bytes,
        media_type: str,
        kind: str,
    ) -> AssetRecord:
        digest = hashlib.sha256(data).hexdigest()
        write_atomic(self.output_root / relative_path, data)
        return AssetRecord(relative_path, digest, len(data), media_type, kind, None)


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def _image_payload(mime: str, value: Any) -> tuple[bytes, str]:
    if mime in {"image/png", "image/jpeg"}:
        if isinstance(value, bytes):
            return value, ".png" if mime == "image/png" else ".jpg"
        return base64.b64decode(_as_text(value)), ".png" if mime == "image/png" else ".jpg"
    return _as_text(value).encode("utf-8"), ".svg"


def _metadata(*, kind: str, record: AssetRecord, **extra: str) -> dict[str, Any]:
    return {
        "kind": kind,
        **extra,
        "asset_href": f"../{record.relative_path.as_posix()}",
        "size_bytes": record.size_bytes,
    }


def prepare_notebook_assets(
    analysis: NotebookAnalysis,
    store: AssetStore,
) -> PreparedNotebook:
    notebook = copy.deepcopy(analysis.notebook)
    notebook_key = analysis.source.stem
    namespace = f"notebooks/{notebook_key}"
    assets_by_path: dict[Path, AssetRecord] = {}

    for cell_index, cell in enumerate(notebook.cells):
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            output_metadata = output.setdefault("metadata", {})
            chart_id = f"{notebook_key}-cell-{cell_index}-output-{output_index}"

            if PLOTLY_MIME in data:
                plotly_json = _compact_json_bytes(data[PLOTLY_MIME])
                payload_digest = hashlib.sha256(plotly_json).hexdigest()
                payload_key = f"plotly-{payload_digest}"
                script = (
                    b"window.UnfallatlasPresentation.registerPlotlyPayload("
                    + _compact_json_bytes(payload_key)
                    + b","
                    + plotly_json
                    + b");\n"
                )
                record = store.put_bytes(
                    namespace=namespace,
                    stem="plotly",
                    suffix=".js",
                    data=script,
                    media_type="text/javascript",
                    kind="plotly",
                    cell_index=cell_index,
                )
                assets_by_path.setdefault(record.relative_path, record)
                output_metadata["unfallatlas_presentation"] = _metadata(
                    kind="plotly",
                    record=record,
                    chart_id=chart_id,
                    payload_key=payload_key,
                )
                continue

            for mime in ("image/svg+xml", "image/png", "image/jpeg"):
                if mime not in data:
                    continue
                image_bytes, suffix = _image_payload(mime, data[mime])
                record = store.put_bytes(
                    namespace=namespace,
                    stem="image",
                    suffix=suffix,
                    data=image_bytes,
                    media_type=mime,
                    kind="image",
                    cell_index=cell_index,
                )
                assets_by_path.setdefault(record.relative_path, record)
                output_metadata["unfallatlas_presentation"] = _metadata(kind="image", record=record)
                break

    return PreparedNotebook(notebook=notebook, assets=tuple(assets_by_path.values()))


def copy_shared_assets(store: AssetStore) -> tuple[AssetRecord, ...]:
    plotly_resource = resources.files("plotly") / "package_data" / "plotly.min.js"
    plotly_record = store.put_named_bytes(
        relative_path=Path("assets/vendor") / f"plotly-{plotly.__version__}.min.js",
        data=plotly_resource.read_bytes(),
        media_type="text/javascript",
        kind="plotly-runtime",
    )
    mathjax_record = store.put_named_bytes(
        relative_path=Path("assets/vendor") / MATHJAX_FILENAME,
        data=(VENDOR_DIR / MATHJAX_FILENAME).read_bytes(),
        media_type="text/javascript",
        kind="mathjax-runtime",
    )
    css_record = store.put_bytes(
        namespace="ui",
        stem="presentation",
        suffix=".css",
        data=(STATIC_DIR / "presentation.css").read_bytes(),
        media_type="text/css",
        kind="ui-style",
        cell_index=None,
    )
    javascript_record = store.put_bytes(
        namespace="ui",
        stem="presentation",
        suffix=".js",
        data=(STATIC_DIR / "presentation.js").read_bytes(),
        media_type="text/javascript",
        kind="ui-script",
        cell_index=None,
    )
    return plotly_record, mathjax_record, css_record, javascript_record
