"""Loading of the SROIE ground-truth split.

Every defect found here stops the run. A benchmark that quietly drops
records reports a number for a corpus nobody chose: a missing image scores
a guaranteed miss against the model, and a missing entity file shrinks the
denominator so the remaining scores look better than they are.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SROIE_FIELDS = ("company", "date", "address", "total")

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


class SroieDatasetError(ValueError):
    """The SROIE split on disk is not usable as a benchmark input."""


@dataclass(frozen=True)
class SroieRecord:
    """One receipt: its image and the four ground-truth values, as written.

    Values are kept exactly as the corpus writes them. Canonicalisation
    belongs at scoring time, where both sides go through the same rules.
    """

    image_id: str
    image_path: Path
    company: str
    date: str
    address: str
    total: str


def _fail(what: str, where: Path, expected: str, how_to_fix: str) -> None:
    """Raise a dataset error carrying all four diagnostic elements."""
    raise SroieDatasetError(f"What: {what}\nWhere: {where}\nExpected: {expected}\nHow to fix: {how_to_fix}")


def _image_index(image_dir: Path) -> dict[str, Path]:
    """Map image id to path for every image file in the directory."""
    return {
        path.stem: path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in _IMAGE_SUFFIXES
    }


def load_sroie_split(split_dir: Path) -> list[SroieRecord]:
    """Load every record in a SROIE split directory.

    Args:
        split_dir: Directory holding ``img/`` and ``entities/``.

    Returns:
        Records sorted by image id, one per entity file.

    Raises:
        SroieDatasetError: If the directories are missing, empty, or the
            images and entity files do not correspond exactly.
    """
    entity_dir = split_dir / "entities"
    image_dir = split_dir / "img"

    for required in (entity_dir, image_dir):
        if not required.is_dir():
            _fail(
                what=f"required directory {required.name!r} is missing.",
                where=required,
                expected="a SROIE split holding both 'img/' and 'entities/'.",
                how_to_fix=(
                    "point pipeline.sroie.data_dir in config/run_config.yml at "
                    "the split root (the directory CONTAINING img/ and "
                    "entities/), not at either subdirectory."
                ),
            )

    entity_paths = sorted(entity_dir.glob("*.txt"))
    if not entity_paths:
        _fail(
            what="no entity files found, so there is nothing to score.",
            where=entity_dir,
            expected="one '<image_id>.txt' per receipt (347 in the test split).",
            how_to_fix=(
                "check pipeline.sroie.data_dir in config/run_config.yml points at a populated split."
            ),
        )

    images = _image_index(image_dir)
    entity_ids = {path.stem for path in entity_paths}

    missing_images = sorted(entity_ids - images.keys())
    if missing_images:
        _fail(
            what=(f"{len(missing_images)} entity file(s) have no image: {', '.join(missing_images[:5])}."),
            where=image_dir,
            expected="every '<image_id>.txt' to have a matching image file.",
            how_to_fix=(
                "re-copy the split so img/ and entities/ correspond exactly. "
                "Do NOT skip these records — scoring a receipt the model never "
                "saw counts a certain miss as a model failure."
            ),
        )

    missing_entities = sorted(images.keys() - entity_ids)
    if missing_entities:
        _fail(
            what=(
                f"{len(missing_entities)} image(s) have no ground truth: {', '.join(missing_entities[:5])}."
            ),
            where=entity_dir,
            expected="every image to have a matching '<image_id>.txt'.",
            how_to_fix=(
                "re-copy the split so img/ and entities/ correspond exactly. "
                "Do NOT skip these images — an unscoreable image silently "
                "shrinks the denominator."
            ),
        )

    records = []
    for entity_path in entity_paths:
        image_id = entity_path.stem
        try:
            payload = json.loads(entity_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            _fail(
                what=f"entity file {entity_path.name!r} is not valid JSON ({err}).",
                where=entity_path,
                expected=(
                    'a JSON object, e.g. {"company": "...", "date": '
                    '"15/01/2019", "address": "...", "total": "193.00"}.'
                ),
                how_to_fix="repair or re-copy that file from the SROIE archive.",
            )

        # Absent and blank are the same defect: no answer to score against.
        # One train-split record ships an empty 'total'.
        missing_fields = [
            field for field in SROIE_FIELDS if field not in payload or not str(payload[field]).strip()
        ]
        if missing_fields:
            _fail(
                what=(f"record {image_id!r} has missing or blank field(s): {', '.join(missing_fields)}."),
                where=entity_path,
                expected=f"all four SROIE keys, each non-empty: {', '.join(SROIE_FIELDS)}.",
                how_to_fix=(
                    "repair or re-copy that file. Do NOT default the value to "
                    "NOT_FOUND — an absent answer key is not a wrong answer, "
                    "and scoring against it would charge the model for a "
                    "question the corpus never asked."
                ),
            )

        records.append(
            SroieRecord(
                image_id=image_id,
                image_path=images[image_id],
                company=payload["company"],
                date=payload["date"],
                address=payload["address"],
                total=payload["total"],
            )
        )
    return records
