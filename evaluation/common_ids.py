from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from evaluation.artifacts import write_json_create_only
from evaluation.io import iter_jsonl
from evaluation.manifest import is_successful_generation, sample_id
from evaluation.metric_contracts import sha256_manifest, sha256_sample_id_set


class CommonIdError(ValueError):
    """Raised when a common evaluation ID set cannot be derived safely."""


@dataclass(frozen=True)
class ManifestIdSet:
    path: Path
    manifest_sha256: str
    ordered_ids: tuple[str, ...]
    successful_ids: frozenset[str]


def inspect_manifest_ids(path: Path) -> ManifestIdSet:
    rows = [item.record for item in iter_jsonl(path)]
    if not rows:
        raise CommonIdError(f"manifest contains no rows: {path}")
    ordered: list[str] = []
    successful: set[str] = set()
    for ordinal, row in enumerate(rows):
        identifier = sample_id(row, ordinal)
        if identifier in ordered:
            raise CommonIdError(f"manifest contains duplicate sample ID {identifier!r}: {path}")
        ordered.append(identifier)
        if is_successful_generation(row):
            successful.add(identifier)
    return ManifestIdSet(
        path=path,
        manifest_sha256=sha256_manifest(path),
        ordered_ids=tuple(ordered),
        successful_ids=frozenset(successful),
    )


def derive_common_successful_ids(manifest_paths: Sequence[Path]) -> dict[str, Any]:
    if not manifest_paths:
        raise CommonIdError("at least one manifest is required")
    manifests = tuple(inspect_manifest_ids(Path(path)) for path in manifest_paths)
    common = set(manifests[0].successful_ids)
    for manifest in manifests[1:]:
        common.intersection_update(manifest.successful_ids)
    ordered_common = tuple(identifier for identifier in manifests[0].ordered_ids if identifier in common)
    if not ordered_common:
        raise CommonIdError("manifests have no common successful sample IDs")
    return {
        "schema_version": 1,
        "policy": "intersection_of_successful_generation_rows",
        "canonical_order_manifest": str(manifests[0].path),
        "common_sample_ids": list(ordered_common),
        "common_sample_count": len(ordered_common),
        "common_sample_id_set_sha256": sha256_sample_id_set(ordered_common),
        "source_manifests": [
            {
                "path": str(manifest.path),
                "manifest_sha256": manifest.manifest_sha256,
                "row_count": len(manifest.ordered_ids),
                "successful_row_count": len(manifest.successful_ids),
                "excluded_from_common_count": len(manifest.successful_ids - common),
            }
            for manifest in manifests
        ],
    }


def write_common_successful_ids(manifest_paths: Sequence[Path], output_path: Path) -> dict[str, Any]:
    artifact = derive_common_successful_ids(manifest_paths)
    write_json_create_only(output_path, artifact)
    return artifact
