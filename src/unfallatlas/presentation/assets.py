from __future__ import annotations

import base64
import copy
import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import plotly
from nbformat import NotebookNode

from unfallatlas.presentation.models import AssetRecord, NotebookAnalysis
from unfallatlas.presentation.validation import WIDGET_VIEW_MIME, select_widget_static_fallback

PACKAGE_DIR = Path(__file__).parent
STATIC_DIR = PACKAGE_DIR / "static"
VENDOR_DIR = PACKAGE_DIR / "vendor"
FONTS_DIR = VENDOR_DIR / "fonts"
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

    def _target_for(self, relative_path: Path) -> Path:
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("asset path must be a relative path inside output_root")
        output_root = self.output_root.resolve(strict=False)
        target = (output_root / relative_path).resolve(strict=False)
        if not target.is_relative_to(output_root):
            raise ValueError("asset path must be a relative path inside output_root")
        return target

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
        for component in (namespace, stem, suffix):
            path_component = Path(component)
            if path_component.is_absolute() or ".." in path_component.parts:
                raise ValueError("asset path must be a relative path inside output_root")
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("assets") / namespace / f"{stem}-{digest[:16]}{suffix}"
        write_atomic(self._target_for(relative), data)
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
        write_atomic(self._target_for(relative_path), data)
        return AssetRecord(relative_path, digest, len(data), media_type, kind, None)

    def prune_namespace(self, namespace: str, keep: tuple[Path, ...]) -> None:
        namespace_path = Path("assets") / namespace
        namespace_root = self._target_for(namespace_path)
        keep_targets = {
            self._target_for(relative_path)
            for relative_path in keep
            if relative_path.is_relative_to(namespace_path)
        }
        if not namespace_root.is_dir():
            return

        for candidate in namespace_root.rglob("*"):
            if candidate.is_file() and candidate.resolve(strict=False) not in keep_targets:
                candidate.unlink()
        for directory in sorted(
            namespace_root.rglob("*"),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


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


def _metadata(
    *,
    kind: str,
    record: AssetRecord,
    href_prefix: Path = Path(".."),
    **extra: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        **extra,
        "asset_href": asset_href(record.relative_path, href_prefix),
        "size_bytes": record.size_bytes,
    }


def asset_href(relative_path: Path, href_prefix: Path = Path("..")) -> str:
    """Return a portable URL whose individual filesystem path segments are encoded."""
    path = href_prefix / relative_path
    return "/".join(quote(part, safe="-._~") for part in path.parts)


def _repository_root(source: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root.resolve(strict=False)
    source_path = source.resolve(strict=False)
    for candidate in (source_path.parent, *source_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate
    raise ValueError(f"cannot determine repository root for local resources in {source}")


def _publish_plotly_layout_images(
    payload: Any,
    *,
    analysis: NotebookAnalysis,
    store: AssetStore,
    repo_root: Path | None,
    href_prefix: Path,
    namespace: str,
    cell_index: int,
) -> tuple[AssetRecord, ...]:
    if not isinstance(payload, dict):
        return ()
    layout = payload.get("layout")
    if not isinstance(layout, dict) or not isinstance(layout.get("images"), list):
        return ()

    records: dict[Path, AssetRecord] = {}
    for image in layout["images"]:
        if not isinstance(image, dict) or not isinstance(image.get("source"), str):
            continue
        reference = image["source"].strip()
        parsed = urlsplit(reference)
        if (
            not reference
            or reference.startswith("#")
            or reference.startswith("//")
            or parsed.scheme
            or not parsed.path
        ):
            continue
        root = _repository_root(analysis.source, repo_root)
        source = (analysis.source.parent / unquote(parsed.path)).resolve(strict=False)
        if not source.is_relative_to(root):
            raise ValueError(f"local resource escapes repository: {reference}")
        if not source.is_file():
            raise FileNotFoundError(f"local resource does not exist: {reference}")
        media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        record = store.put_bytes(
            namespace=f"{namespace}/local",
            stem=f"resource-{source.stem}",
            suffix=source.suffix,
            data=source.read_bytes(),
            media_type=media_type,
            kind="local-resource",
            cell_index=cell_index,
        )
        records.setdefault(record.relative_path, record)
        image["source"] = urlunsplit(
            (
                "",
                "",
                asset_href(record.relative_path, href_prefix),
                parsed.query,
                parsed.fragment,
            )
        )
    return tuple(records.values())


def prepare_notebook_assets(
    analysis: NotebookAnalysis,
    store: AssetStore,
    *,
    href_prefix: Path = Path(".."),
    repo_root: Path | None = None,
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

            if WIDGET_VIEW_MIME in data:
                fallback = select_widget_static_fallback(data)
                fallback_value = data.get(fallback) if fallback is not None else None
                data.clear()
                if fallback is not None:
                    data[fallback] = fallback_value

            if PLOTLY_MIME in data:
                local_records = _publish_plotly_layout_images(
                    data[PLOTLY_MIME],
                    analysis=analysis,
                    store=store,
                    repo_root=repo_root,
                    href_prefix=href_prefix,
                    namespace=namespace,
                    cell_index=cell_index,
                )
                for local_record in local_records:
                    assets_by_path.setdefault(local_record.relative_path, local_record)
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
                    href_prefix=href_prefix,
                    chart_id=chart_id,
                    payload_key=payload_key,
                )
                del data[PLOTLY_MIME]
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
                output_metadata["unfallatlas_presentation"] = _metadata(
                    kind="image", record=record, href_prefix=href_prefix
                )
                del data[mime]
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
    # Fonts keep their stable names (no content hash) because presentation.css
    # references them by relative URL; the CSS itself is content-hashed, so a
    # font change still propagates through the CSS digest that embeds its URLs.
    font_records = tuple(
        store.put_named_bytes(
            relative_path=Path("assets/vendor/fonts") / font_path.name,
            data=font_path.read_bytes(),
            media_type="font/woff2",
            kind="ui-font",
        )
        for font_path in sorted(FONTS_DIR.glob("*.woff2"))
    )
    return plotly_record, mathjax_record, css_record, javascript_record, *font_records
