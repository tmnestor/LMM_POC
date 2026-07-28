"""Give every document in an evaluation set a unique key.

The generator emits ``synthetic.yml`` with one ``CASE0NN`` key per *case*, repeated
once per document type. YAML allows duplicate keys, so a plain ``yaml.safe_load``
silently keeps only the last block and drops two thirds of the ground truth. This
script rewrites both the YAML and the image filenames so that:

    CASE001 (layout: cba_standard)   -> CASE001_bank_statement
    CASE001 (layout: tax_invoice_..) -> CASE001_invoice
    CASE001 (layout: receipt_fuel)   -> CASE001_receipt

Every key is then unique, ``safe_load`` is safe, and the key matches the image
stem exactly. The original ``layout`` value is preserved inside each block, so no
information is lost.

Defaults to a dry run. Nothing is touched without ``--apply``.

Usage:
    python scripts/relabel_evaluation_set.py                    # preview
    python scripts/relabel_evaluation_set.py --apply            # rename
    python scripts/relabel_evaluation_set.py --apply --dir /some/other/set
"""

import argparse
import csv
import shutil
from pathlib import Path

import yaml

DEFAULT_DIR = Path("/Users/tod/Desktop/evaluation_data/synthetic_20260728")
YAML_NAME = "synthetic.yml"

DOC_TYPE_SUFFIX = {
    "BANK_STATEMENT": "bank_statement",
    "INVOICE": "invoice",
    "RECEIPT": "receipt",
}


def load_blocks(yaml_path: Path) -> list[tuple[str, dict]]:
    """Parse the YAML preserving duplicate top-level keys."""
    text = yaml_path.read_text()
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if root is None:
        raise SystemExit(
            f"❌ FATAL: {yaml_path} is empty or not a YAML mapping.\n"
            f"   Expected: a mapping of CASE ids to document blocks, e.g.\n"
            f"       CASE001:\n"
            f"         layout: cba_standard\n"
            f"         fields:\n"
            f"           DOCUMENT_TYPE: BANK_STATEMENT\n"
            f"   Recover: point --dir at an evaluation set containing a valid {YAML_NAME}."
        )

    loader = yaml.SafeLoader(text)
    return [(key.value, loader.construct_document(value)) for key, value in root.value]


def plan_renames(blocks: list[tuple[str, dict]], eval_dir: Path) -> list[dict]:
    """Work out the old -> new name for every document, validating as we go."""
    plan = []
    seen_keys: dict[str, str] = {}

    for case_id, block in blocks:
        missing = [k for k in ("layout", "fields") if k not in block]
        if missing:
            raise SystemExit(
                f"❌ FATAL: block '{case_id}' is missing required key(s): {missing}.\n"
                f"   Where: {eval_dir / YAML_NAME}, under '{case_id}:'\n"
                f"   Expected: every block needs 'layout' and 'fields', e.g.\n"
                f"       {case_id}:\n"
                f"         layout: cba_standard\n"
                f"         fields:\n"
                f"           DOCUMENT_TYPE: BANK_STATEMENT\n"
                f"   Recover: add the missing key(s), or regenerate the evaluation set."
            )

        doc_type = block["fields"].get("DOCUMENT_TYPE")
        if doc_type not in DOC_TYPE_SUFFIX:
            raise SystemExit(
                f"❌ FATAL: block '{case_id}' has DOCUMENT_TYPE={doc_type!r}, "
                f"which has no filename suffix mapping.\n"
                f"   Where: {eval_dir / YAML_NAME}, at '{case_id}.fields.DOCUMENT_TYPE'\n"
                f"   Allowed values: {sorted(DOC_TYPE_SUFFIX)}\n"
                f"   Recover: correct DOCUMENT_TYPE, or add a mapping for it to "
                f"DOC_TYPE_SUFFIX in {Path(__file__).name}."
            )

        new_key = f"{case_id}_{DOC_TYPE_SUFFIX[doc_type]}"
        old_stem = f"{case_id}_{block['layout']}"

        if new_key in seen_keys:
            raise SystemExit(
                f"❌ FATAL: key collision - '{new_key}' would be produced twice "
                f"(from '{seen_keys[new_key]}' and from '{old_stem}').\n"
                f"   Where: {eval_dir / YAML_NAME}\n"
                f"   Cause: case '{case_id}' has two documents of type {doc_type}.\n"
                f"   Expected: exactly one bank_statement, one invoice and one receipt "
                f"per case.\n"
                f"   Recover: remove or re-type the duplicate block before relabelling."
            )
        seen_keys[new_key] = old_stem

        old_image = eval_dir / f"{old_stem}.png"
        if not old_image.exists():
            raise SystemExit(
                f"❌ FATAL: no image on disk for block '{case_id}' (layout "
                f"'{block['layout']}').\n"
                f"   Expected file: {old_image}\n"
                f"   Recover: restore the missing image, or remove the orphaned block "
                f"from {YAML_NAME}."
            )

        plan.append(
            {
                "old_key": case_id,
                "new_key": new_key,
                "layout": block["layout"],
                "document_type": doc_type,
                "old_image": old_image,
                "new_image": eval_dir / f"{new_key}.png",
                "block": block,
            }
        )

    return plan


def check_unclaimed_images(plan: list[dict], eval_dir: Path) -> list[Path]:
    claimed = {item["old_image"] for item in plan}
    return sorted(p for p in eval_dir.glob("*.png") if p not in claimed)


def write_yaml(plan: list[dict], yaml_path: Path) -> None:
    """Rewrite the YAML with unique keys, preserving field order and content."""
    document = {}
    for item in plan:
        block = dict(item["block"])
        # Record what the key used to be so the rename stays traceable.
        block["source_case"] = item["old_key"]
        document[item["new_key"]] = block

    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=10**6))


def verify(plan: list[dict], yaml_path: Path, eval_dir: Path) -> None:
    """Prove the rewrite is lossless: safe_load must now see every document."""
    reloaded = yaml.safe_load(yaml_path.read_text())

    if len(reloaded) != len(plan):
        raise SystemExit(
            f"❌ FATAL: verification failed - safe_load returned {len(reloaded)} "
            f"entries but {len(plan)} were written.\n"
            f"   Where: {yaml_path}\n"
            f"   Cause: duplicate keys survived the rewrite.\n"
            f"   Recover: restore {yaml_path.name}.bak and re-run."
        )

    for item in plan:
        key = item["new_key"]
        if reloaded[key]["fields"] != item["block"]["fields"]:
            raise SystemExit(
                f"❌ FATAL: verification failed - fields changed for '{key}'.\n"
                f"   Where: {yaml_path}\n"
                f"   Recover: restore {yaml_path.name}.bak and re-run."
            )
        if not item["new_image"].exists():
            raise SystemExit(
                f"❌ FATAL: verification failed - expected image {item['new_image']} "
                f"is missing after rename.\n"
                f"   Recover: consult relabel_mapping.csv in {eval_dir} to undo."
            )

    print(
        f"✅ verified: safe_load now returns all {len(reloaded)} documents, "
        f"fields unchanged, every image present"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dir", type=Path, default=DEFAULT_DIR, help=f"evaluation set directory (default: {DEFAULT_DIR})"
    )
    parser.add_argument(
        "--apply", action="store_true", help="actually rename; without this it is a dry run"
    )
    args = parser.parse_args()

    eval_dir: Path = args.dir
    yaml_path = eval_dir / YAML_NAME
    if not yaml_path.exists():
        raise SystemExit(
            f"❌ FATAL: {YAML_NAME} not found.\n"
            f"   Looked in: {eval_dir}\n"
            f"   Recover: pass --dir pointing at the evaluation set directory."
        )

    blocks = load_blocks(yaml_path)
    plan = plan_renames(blocks, eval_dir)

    unclaimed = check_unclaimed_images(plan, eval_dir)
    duplicate_keys = len(blocks) - len({key for key, _ in blocks})

    print(f"evaluation set : {eval_dir}")
    print(f"documents      : {len(plan)}")
    print(f"unique keys    : before {len({k for k, _ in blocks})}, after {len(plan)}")
    print(f"duplicate keys : {duplicate_keys} (these are what safe_load was dropping)")
    if unclaimed:
        print(f"⚠️  images not referenced by the YAML ({len(unclaimed)}): {[p.name for p in unclaimed[:5]]}")

    print("\nsample renames:")
    for item in plan[:6]:
        print(f"  {item['old_image'].name:<42} -> {item['new_image'].name}")
    print(f"  ... {len(plan) - 6} more")

    if not args.apply:
        print("\nDRY RUN - nothing changed. Re-run with --apply to perform the rename.")
        return

    # Back up the YAML before touching anything.
    backup = yaml_path.with_suffix(".yml.bak")
    shutil.copy2(yaml_path, backup)
    print(f"\nbacked up {yaml_path.name} -> {backup.name}")

    # Record the mapping first so the rename is always reversible.
    mapping_path = eval_dir / "relabel_mapping.csv"
    with mapping_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["old_key", "new_key", "old_image", "new_image", "layout", "document_type"])
        for item in plan:
            writer.writerow(
                [
                    item["old_key"],
                    item["new_key"],
                    item["old_image"].name,
                    item["new_image"].name,
                    item["layout"],
                    item["document_type"],
                ]
            )
    print(f"wrote {mapping_path.name}")

    # Two-phase rename so a new name can never clobber a not-yet-renamed file.
    for index, item in enumerate(plan):
        item["old_image"].rename(eval_dir / f".relabel_tmp_{index}.png")
    for index, item in enumerate(plan):
        (eval_dir / f".relabel_tmp_{index}.png").rename(item["new_image"])
    print(f"renamed {len(plan)} images")

    write_yaml(plan, yaml_path)
    print(f"rewrote {yaml_path.name} with unique keys")

    verify(plan, yaml_path, eval_dir)


if __name__ == "__main__":
    main()
