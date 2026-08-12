"""Curated, scene-independent visual object reference catalog.

The editable source lives below ``assets/object_references`` and is deliberately
kept outside git.  A human curates crops and three representative views in the
offline editor; :func:`build_object_reference_catalog` then publishes an
immutable SQLite/image bundle.  Scene QA only reads a completed bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Iterable
import unicodedata

from PIL import Image, ImageOps


MANIFEST_SCHEMA = "roomie.object_references.manifest.v1"
DATABASE_SCHEMA = "roomie.object_references.sqlite.v1"
DEFAULT_IMAGE_COUNT = 3
DEFAULT_IMAGE_MAX_SIDE_PX = 768
DEFAULT_JPEG_QUALITY = 85
_BUILD_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1, including the ROS 2 Humble system package.
    _LANCZOS = Image.LANCZOS


class ObjectReferenceError(RuntimeError):
    """Raised when an editable or published reference catalog is invalid."""


def default_object_reference_root() -> Path:
    """Locate the ignored source-tree data directory from source or an install."""

    configured = os.environ.get("ROOMIE_OBJECT_REFERENCES_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    here = Path(__file__).resolve()
    direct = here.parents[2] / "assets" / "object_references"
    candidates = [direct]
    for parent in here.parents:
        candidates.append(parent / "src" / "roomie" / "assets" / "object_references")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return direct.resolve()


@dataclass(frozen=True)
class PublishedReferenceImage:
    image_id: str
    view_label: str
    rank: int
    path: Path
    width: int
    height: int
    byte_count: int
    sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "view_label": self.view_label,
            "rank": self.rank,
            "width": self.width,
            "height": self.height,
            "bytes": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PublishedObjectReference:
    reference_id: str
    name: str
    aliases: tuple[str, ...]
    images: tuple[PublishedReferenceImage, ...]

    def metadata(self, *, include_images: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "reference_id": self.reference_id,
            "name": self.name,
            "aliases": list(self.aliases),
        }
        if include_images:
            value["images"] = [image.metadata() for image in self.images]
        return value


@dataclass(frozen=True)
class ObjectReferenceCatalogLoad:
    root: Path
    catalog: "ObjectReferenceCatalog | None"
    status: str

    @property
    def available(self) -> bool:
        return self.catalog is not None


class ObjectReferenceCatalog:
    """Fully validated in-memory view of one immutable published build."""

    def __init__(
        self,
        root: Path,
        build_id: str,
        build_root: Path,
        references: Iterable[PublishedObjectReference],
    ):
        self.root = root
        self.build_id = build_id
        self.build_root = build_root
        self._references = {
            reference.reference_id: reference for reference in references
        }
        if not self._references:
            raise ObjectReferenceError("published catalog contains no references")

    @classmethod
    def load(cls, root: str | Path) -> "ObjectReferenceCatalog":
        catalog_root = Path(root).expanduser().resolve()
        current_path = catalog_root / "processed" / "CURRENT"
        if not current_path.is_file():
            raise ObjectReferenceError(f"published pointer does not exist: {current_path}")
        build_id = current_path.read_text(encoding="utf-8").strip()
        if not _BUILD_ID.fullmatch(build_id):
            raise ObjectReferenceError("published pointer contains an invalid build id")
        builds_root = (catalog_root / "processed" / "builds").resolve()
        build_root = (builds_root / build_id).resolve()
        if build_root.parent != builds_root or not build_root.is_dir():
            raise ObjectReferenceError(
                f"published build directory does not exist: {build_root}"
            )
        database_path = build_root / "catalog.sqlite3"
        if not database_path.is_file():
            raise ObjectReferenceError(
                f"published database does not exist: {database_path}"
            )

        connection = sqlite3.connect(
            f"{database_path.as_uri()}?mode=ro&immutable=1", uri=True
        )
        connection.row_factory = sqlite3.Row
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or str(integrity[0]).lower() != "ok":
                raise ObjectReferenceError("published database failed integrity_check")
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            if metadata.get("schema_version") != DATABASE_SCHEMA:
                raise ObjectReferenceError(
                    "unsupported published database schema: "
                    f"{metadata.get('schema_version')!r}"
                )
            if metadata.get("build_id") != build_id:
                raise ObjectReferenceError("CURRENT and database build ids disagree")

            image_rows: dict[str, list[PublishedReferenceImage]] = {}
            for row in connection.execute(
                """
                SELECT reference_id, image_id, view_label, rank, relative_path,
                       width, height, byte_count, sha256
                FROM reference_images
                ORDER BY reference_id, rank, image_id
                """
            ):
                relative_path = Path(str(row["relative_path"]))
                image_path = (build_root / relative_path).resolve()
                if build_root not in image_path.parents or not image_path.is_file():
                    raise ObjectReferenceError(
                        f"published image is missing or escapes its build: {relative_path}"
                    )
                expected_bytes = int(row["byte_count"])
                if image_path.stat().st_size != expected_bytes:
                    raise ObjectReferenceError(
                        f"published image size mismatch: {relative_path}"
                    )
                expected_hash = str(row["sha256"])
                if _sha256_file(image_path) != expected_hash:
                    raise ObjectReferenceError(
                        f"published image hash mismatch: {relative_path}"
                    )
                image_rows.setdefault(str(row["reference_id"]), []).append(
                    PublishedReferenceImage(
                        image_id=str(row["image_id"]),
                        view_label=str(row["view_label"]),
                        rank=int(row["rank"]),
                        path=image_path,
                        width=int(row["width"]),
                        height=int(row["height"]),
                        byte_count=expected_bytes,
                        sha256=expected_hash,
                    )
                )

            references = []
            for row in connection.execute(
                "SELECT reference_id, name, aliases_json FROM object_references "
                "ORDER BY reference_id"
            ):
                reference_id = str(row["reference_id"])
                try:
                    aliases_value = json.loads(str(row["aliases_json"]))
                except json.JSONDecodeError as exc:
                    raise ObjectReferenceError(
                        f"invalid aliases for reference {reference_id}"
                    ) from exc
                if not isinstance(aliases_value, list) or not all(
                    isinstance(value, str) for value in aliases_value
                ):
                    raise ObjectReferenceError(
                        f"invalid aliases for reference {reference_id}"
                    )
                images = tuple(image_rows.get(reference_id, ()))
                if len(images) != DEFAULT_IMAGE_COUNT or [
                    image.rank for image in images
                ] != list(range(1, DEFAULT_IMAGE_COUNT + 1)):
                    raise ObjectReferenceError(
                        f"reference {reference_id} must publish exactly three ranked images"
                    )
                references.append(
                    PublishedObjectReference(
                        reference_id=reference_id,
                        name=str(row["name"]),
                        aliases=tuple(aliases_value),
                        images=images,
                    )
                )
        except sqlite3.Error as exc:
            raise ObjectReferenceError(f"could not read published database: {exc}") from exc
        finally:
            connection.close()
        return cls(catalog_root, build_id, build_root, references)

    def get(self, reference_id: str) -> PublishedObjectReference:
        key = str(reference_id).strip()
        try:
            return self._references[key]
        except KeyError as exc:
            raise KeyError(f"unknown reference_id: {reference_id}") from exc

    def references(self) -> list[PublishedObjectReference]:
        return [self._references[key] for key in sorted(self._references)]

    def inventory(self) -> list[dict[str, Any]]:
        return [
            reference.metadata(include_images=False) for reference in self.references()
        ]


def try_load_object_reference_catalog(
    root: str | Path,
) -> ObjectReferenceCatalogLoad:
    resolved = Path(root).expanduser().resolve()
    try:
        catalog = ObjectReferenceCatalog.load(resolved)
    except Exception as exc:
        return ObjectReferenceCatalogLoad(
            root=resolved,
            catalog=None,
            status=f"{type(exc).__name__}: {exc}",
        )
    return ObjectReferenceCatalogLoad(
        root=resolved,
        catalog=catalog,
        status=(
            f"available build={catalog.build_id} "
            f"references={len(catalog.references())}"
        ),
    )


def reference_catalog_prompt(load: ObjectReferenceCatalogLoad) -> str:
    """Return runtime-only system prompt text matching actual tool availability."""

    if load.catalog is None:
        return (
            "Object-reference image tools are unavailable for this run. "
            "Do not claim that a scene object was visually compared with a local "
            f"reference catalog. Runtime status: {load.status}"
        )
    lines = [
        "A curated local object-reference catalog is available for visual identity "
        "checks. It is independent of the scene graph.",
        "Known references (use the exact reference_id in tools):",
    ]
    for reference in load.catalog.references():
        aliases = ", ".join(reference.aliases)
        suffix = f"; aliases: {aliases}" if aliases else ""
        lines.append(
            f"- reference_id={reference.reference_id!r}; name={reference.name!r}{suffix}"
        )
    lines.extend(
        [
            "Use inspect_object_reference to view one reference alone. Once concrete "
            "scene object ids are known, use compare_scene_objects_to_reference to "
            "attach those object snapshots and the chosen reference views together.",
            "The comparison tool does not search, classify, OCR, or decide identity; "
            "make the visual judgment yourself from the returned evidence. If candidate "
            "snapshots are missing or inconclusive, say that the images are insufficient.",
            "Reference tools do not change scene scope: the stable scene graph is not a "
            "strict statement about the robot's current camera field of view.",
        ]
    )
    return "\n".join(lines)


def empty_object_reference_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA,
        "settings": {
            "image_count": DEFAULT_IMAGE_COUNT,
            "image_max_side_px": DEFAULT_IMAGE_MAX_SIDE_PX,
            "jpeg_quality": DEFAULT_JPEG_QUALITY,
        },
        "references": [],
    }


def load_or_scan_object_reference_manifest(root: str | Path) -> dict[str, Any]:
    """Load the editable manifest and merge newly discovered raw images into it."""

    catalog_root = Path(root).expanduser().resolve()
    manifest_path = catalog_root / "catalog.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ObjectReferenceError(f"invalid editable manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ObjectReferenceError("editable manifest root must be an object")
    else:
        manifest = empty_object_reference_manifest()
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ObjectReferenceError(
            f"unsupported editable manifest schema: {manifest.get('schema_version')!r}"
        )
    references = manifest.get("references")
    if not isinstance(references, list):
        raise ObjectReferenceError("editable manifest references must be an array")

    raw_root = catalog_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    by_id = {
        str(reference.get("reference_id")): reference
        for reference in references
        if isinstance(reference, dict) and reference.get("reference_id")
    }
    for directory in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        reference_id = unicodedata.normalize("NFKC", directory.name).strip()
        reference = by_id.get(reference_id)
        if reference is None:
            reference = {
                "reference_id": reference_id,
                "name": reference_id,
                "aliases": [],
                "images": [],
            }
            references.append(reference)
            by_id[reference_id] = reference
        images = reference.setdefault("images", [])
        if not isinstance(images, list):
            raise ObjectReferenceError(
                f"images for reference {reference_id} must be an array"
            )
        known_sources = {
            str(image.get("source"))
            for image in images
            if isinstance(image, dict) and image.get("source")
        }
        for source_path in sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        ):
            relative = source_path.relative_to(catalog_root).as_posix()
            if relative in known_sources:
                continue
            with Image.open(source_path) as stream:
                width, height = ImageOps.exif_transpose(stream).size
            image_id = _unique_image_id(source_path.stem, images)
            images.append(
                {
                    "image_id": image_id,
                    "source": relative,
                    "source_width": width,
                    "source_height": height,
                    "crop_xyxy": [0, 0, width, height],
                    "crop_confirmed": False,
                    "selected": False,
                    "rank": None,
                    "view_label": "",
                }
            )
    return manifest


def save_object_reference_manifest(
    root: str | Path, manifest: dict[str, Any]
) -> Path:
    catalog_root = Path(root).expanduser().resolve()
    _validate_manifest_shape(catalog_root, manifest, require_publishable=False)
    catalog_root.mkdir(parents=True, exist_ok=True)
    output = catalog_root / "catalog.json"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".catalog.", suffix=".json", dir=catalog_root
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output


def validate_object_reference_manifest(
    root: str | Path,
    manifest: dict[str, Any],
    *,
    require_publishable: bool = True,
) -> list[str]:
    """Return validation messages instead of raising, for use by the editor UI."""

    try:
        _validate_manifest_shape(
            Path(root).expanduser().resolve(),
            manifest,
            require_publishable=require_publishable,
        )
    except ObjectReferenceError as exc:
        return [str(exc)]
    return []


def build_object_reference_catalog(
    root: str | Path,
    manifest: dict[str, Any] | None = None,
) -> ObjectReferenceCatalog:
    """Publish one immutable database/image build and atomically select it."""

    catalog_root = Path(root).expanduser().resolve()
    if manifest is None:
        manifest_path = catalog_root / "catalog.json"
        if not manifest_path.is_file():
            raise ObjectReferenceError(f"editable manifest does not exist: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ObjectReferenceError(f"invalid editable manifest: {exc}") from exc
    _validate_manifest_shape(catalog_root, manifest, require_publishable=True)
    normalized = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    source_hashes = []
    for reference in manifest["references"]:
        for image in reference["images"]:
            if image.get("selected"):
                source_hashes.append(
                    (
                        str(image["source"]),
                        _sha256_file(_source_path(catalog_root, str(image["source"]))),
                    )
                )
    digest = hashlib.sha256(normalized)
    digest.update(
        json.dumps(source_hashes, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    build_id = f"v1-{digest.hexdigest()[:20]}"
    processed_root = catalog_root / "processed"
    builds_root = processed_root / "builds"
    builds_root.mkdir(parents=True, exist_ok=True)
    final_root = builds_root / build_id

    if not final_root.exists():
        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".{build_id}.", dir=builds_root)
        )
        try:
            _write_published_build(temporary_root, build_id, catalog_root, manifest)
            try:
                temporary_root.rename(final_root)
            except FileExistsError:
                shutil.rmtree(temporary_root)
        except Exception:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
            raise

    pointer_fd, pointer_name = tempfile.mkstemp(
        prefix=".CURRENT.", dir=processed_root
    )
    try:
        with os.fdopen(pointer_fd, "w", encoding="utf-8") as stream:
            stream.write(build_id + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pointer_name, processed_root / "CURRENT")
    except Exception:
        try:
            os.unlink(pointer_name)
        except FileNotFoundError:
            pass
        raise
    return ObjectReferenceCatalog.load(catalog_root)


def _validate_manifest_shape(
    root: Path,
    manifest: dict[str, Any],
    *,
    require_publishable: bool,
) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ObjectReferenceError(f"manifest schema must be {MANIFEST_SCHEMA}")
    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise ObjectReferenceError("manifest settings must be an object")
    image_count = _bounded_int(
        settings.get("image_count", DEFAULT_IMAGE_COUNT), "settings.image_count", 1, 8
    )
    if image_count != DEFAULT_IMAGE_COUNT:
        raise ObjectReferenceError("this catalog version requires exactly three images")
    _bounded_int(
        settings.get("image_max_side_px", DEFAULT_IMAGE_MAX_SIDE_PX),
        "settings.image_max_side_px",
        64,
        4096,
    )
    _bounded_int(
        settings.get("jpeg_quality", DEFAULT_JPEG_QUALITY),
        "settings.jpeg_quality",
        30,
        100,
    )
    references = manifest.get("references")
    if not isinstance(references, list) or (require_publishable and not references):
        raise ObjectReferenceError("manifest references must be a non-empty array")
    seen_references: set[str] = set()
    for reference_index, reference in enumerate(references):
        prefix = f"references[{reference_index}]"
        if not isinstance(reference, dict):
            raise ObjectReferenceError(f"{prefix} must be an object")
        reference_id = _checked_text(reference.get("reference_id"), f"{prefix}.reference_id")
        if reference_id in seen_references:
            raise ObjectReferenceError(f"duplicate reference_id: {reference_id}")
        seen_references.add(reference_id)
        _checked_text(reference.get("name"), f"{prefix}.name")
        aliases = reference.get("aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) and alias.strip() for alias in aliases
        ):
            raise ObjectReferenceError(f"{prefix}.aliases must contain non-empty strings")
        images = reference.get("images")
        if not isinstance(images, list):
            raise ObjectReferenceError(f"{prefix}.images must be an array")
        seen_images: set[str] = set()
        selected_ranks: list[int] = []
        for image_index, image in enumerate(images):
            image_prefix = f"{prefix}.images[{image_index}]"
            if not isinstance(image, dict):
                raise ObjectReferenceError(f"{image_prefix} must be an object")
            image_id = _checked_text(image.get("image_id"), f"{image_prefix}.image_id")
            if image_id in seen_images:
                raise ObjectReferenceError(
                    f"duplicate image_id {image_id} in reference {reference_id}"
                )
            seen_images.add(image_id)
            source = _checked_relative_path_text(
                image.get("source"), f"{image_prefix}.source"
            )
            source_path = _source_path(root, source)
            if not source_path.is_file():
                raise ObjectReferenceError(f"raw image does not exist: {source}")
            try:
                with Image.open(source_path) as stream:
                    width, height = ImageOps.exif_transpose(stream).size
            except Exception as exc:
                raise ObjectReferenceError(f"could not read raw image {source}: {exc}") from exc
            crop = image.get("crop_xyxy")
            if not isinstance(crop, list) or len(crop) != 4:
                raise ObjectReferenceError(f"{image_prefix}.crop_xyxy must have four values")
            try:
                left, top, right, bottom = [int(value) for value in crop]
            except (TypeError, ValueError) as exc:
                raise ObjectReferenceError(
                    f"{image_prefix}.crop_xyxy must contain integers"
                ) from exc
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                raise ObjectReferenceError(
                    f"{image_prefix}.crop_xyxy is outside the {width}x{height} image"
                )
            selected = image.get("selected") is True
            if selected:
                try:
                    rank = int(image.get("rank"))
                except (TypeError, ValueError) as exc:
                    raise ObjectReferenceError(
                        f"{image_prefix}.rank must be 1, 2, or 3"
                    ) from exc
                selected_ranks.append(rank)
                if require_publishable and not image.get("crop_confirmed"):
                    raise ObjectReferenceError(
                        f"confirm the crop for selected image {reference_id}/{image_id}"
                    )
                if require_publishable:
                    _checked_text(
                        image.get("view_label"), f"{image_prefix}.view_label"
                    )
        if require_publishable and sorted(selected_ranks) != [1, 2, 3]:
            raise ObjectReferenceError(
                f"reference {reference_id} must select exactly ranks 1, 2, and 3"
            )


def _write_published_build(
    output_root: Path,
    build_id: str,
    catalog_root: Path,
    manifest: dict[str, Any],
) -> None:
    settings = manifest["settings"]
    max_side = int(settings.get("image_max_side_px", DEFAULT_IMAGE_MAX_SIDE_PX))
    jpeg_quality = int(settings.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    database_path = output_root / "catalog.sqlite3"
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE object_references (
              reference_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              aliases_json TEXT NOT NULL
            );
            CREATE TABLE reference_images (
              reference_id TEXT NOT NULL,
              image_id TEXT NOT NULL,
              view_label TEXT NOT NULL,
              rank INTEGER NOT NULL,
              relative_path TEXT NOT NULL,
              width INTEGER NOT NULL,
              height INTEGER NOT NULL,
              byte_count INTEGER NOT NULL,
              sha256 TEXT NOT NULL,
              PRIMARY KEY (reference_id, image_id),
              UNIQUE (reference_id, rank),
              FOREIGN KEY (reference_id) REFERENCES object_references(reference_id)
            );
            """
        )
        connection.execute("PRAGMA foreign_keys=ON")
        metadata = {
            "schema_version": DATABASE_SCHEMA,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "image_max_side_px": str(max_side),
            "jpeg_quality": str(jpeg_quality),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES(?, ?)", metadata.items()
        )
        for reference in manifest["references"]:
            reference_id = str(reference["reference_id"])
            connection.execute(
                "INSERT INTO object_references(reference_id, name, aliases_json) "
                "VALUES(?, ?, ?)",
                (
                    reference_id,
                    str(reference["name"]),
                    json.dumps(reference.get("aliases", []), ensure_ascii=False),
                ),
            )
            selected = sorted(
                (image for image in reference["images"] if image.get("selected")),
                key=lambda image: int(image["rank"]),
            )
            for image in selected:
                source_path = _source_path(catalog_root, str(image["source"]))
                with Image.open(source_path) as stream:
                    processed = ImageOps.exif_transpose(stream).convert("RGB")
                crop = tuple(int(value) for value in image["crop_xyxy"])
                processed = processed.crop(crop)
                processed.thumbnail((max_side, max_side), _LANCZOS)
                asset_name = (
                    hashlib.sha256(
                        f"{reference_id}\0{image['image_id']}".encode("utf-8")
                    ).hexdigest()[:20]
                    + ".jpg"
                )
                asset_path = images_root / asset_name
                processed.save(
                    asset_path,
                    format="JPEG",
                    quality=jpeg_quality,
                    optimize=True,
                    progressive=True,
                )
                relative_path = asset_path.relative_to(output_root).as_posix()
                connection.execute(
                    """
                    INSERT INTO reference_images(
                      reference_id, image_id, view_label, rank, relative_path,
                      width, height, byte_count, sha256
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        reference_id,
                        str(image["image_id"]),
                        str(image["view_label"]),
                        int(image["rank"]),
                        relative_path,
                        processed.width,
                        processed.height,
                        asset_path.stat().st_size,
                        _sha256_file(asset_path),
                    ),
                )
        connection.commit()
    finally:
        connection.close()
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _source_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ObjectReferenceError("raw image paths must be relative to the catalog root")
    candidate = (root / relative_path).resolve()
    raw_root = (root / "raw").resolve()
    if raw_root not in candidate.parents:
        raise ObjectReferenceError(f"raw image path escapes raw/: {relative}")
    return candidate


def _checked_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ObjectReferenceError(f"{label} must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or len(normalized) > 128 or any(ord(char) < 32 for char in normalized):
        raise ObjectReferenceError(f"{label} must be 1-128 printable characters")
    if "/" in normalized or "\\" in normalized:
        raise ObjectReferenceError(f"{label} must not contain path separators")
    return normalized


def _checked_relative_path_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ObjectReferenceError(f"{label} must be a string")
    normalized = unicodedata.normalize("NFKC", value).strip()
    if (
        not normalized
        or len(normalized) > 512
        or any(ord(char) < 32 for char in normalized)
        or "\\" in normalized
    ):
        raise ObjectReferenceError(f"{label} must be a printable relative POSIX path")
    return normalized


def _bounded_int(value: Any, label: str, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ObjectReferenceError(f"{label} must be an integer") from exc
    if parsed < lower or parsed > upper:
        raise ObjectReferenceError(f"{label} must be between {lower} and {upper}")
    return parsed


def _unique_image_id(stem: str, images: list[Any]) -> str:
    base = unicodedata.normalize("NFKC", stem).strip() or "image"
    base = re.sub(r"[\\/\x00-\x1f]", "-", base)[:96]
    used = {
        str(image.get("image_id"))
        for image in images
        if isinstance(image, dict) and image.get("image_id")
    }
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
