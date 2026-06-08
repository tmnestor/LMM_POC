#!/usr/bin/env python3
"""Probe: does batching the linking fallback's VLM calls beat serial?

Background (2026-06-08): the transaction-linking VLM fallback issues its calls
SERIALLY, and ``vllm.defaults.max_num_seqs: 1`` caps the engine to one sequence,
so vLLM's continuous batching never engages. tp=4 serial measured 168s for 33
calls; tp=1 was WORSE (358s). The hypothesis is that the real lever is
concurrency — submit a statement's receipts as ONE batched ``chat`` so vLLM runs
up to ``max_num_seqs`` of them at once (and the shared statement prefix is
computed once). This probe measures that, on the real tp=4 engine, before we
build the feature.

What it does:
  * loads the engine via the prod path (``_build_processor`` → tp from config),
  * builds N single-image linking prompts against ONE bank statement (image-first,
    pre-tiled exactly like the fallback — the per-statement-batch best case),
  * times them SERIALLY (one chat() each) vs BATCHED (one chat() with all N),
  * reports per-call ms and the speedup.

Run ON THE GPU BOX. To test concurrency, set ``vllm.defaults.max_num_seqs`` in
config/run_config.yml (1 = serial-equivalent; try 4 / 8 / 16) BEFORE each run —
the probe prints the active value. If a high max_num_seqs OOMs at engine load,
that is the answer for the safe concurrency ceiling with 19-tile prompts.

Usage:
  python -m scripts.probe_linking_batching --data-dir ../synthetic_transaction_linking --n 33
  python -m scripts.probe_linking_batching --data-dir ../synthetic_transaction_linking \
      --bank-image ../synthetic_transaction_linking/CASE001_bank_statement.png --n 33
"""

import argparse
import time
from datetime import date
from pathlib import Path

from PIL import Image
from vllm import SamplingParams

from common.transaction_matcher import ReceiptSummary
from common.vlm_linker import build_link_prompt, load_link_prompt
from stages.transaction_link import _build_processor

_IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _pick_bank_image(data_dir: Path) -> Path:
    """Pick a bank-statement image (by name hint) or the first image found."""
    imgs = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not imgs:
        raise SystemExit(f"no images found in {data_dir}")
    banks = [p for p in imgs if "bank" in p.name.lower() or "statement" in p.name.lower()]
    return banks[0] if banks else imgs[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--bank-image", type=Path, default=None, help="default: auto-pick from --data-dir")
    ap.add_argument("--n", type=int, default=33, help="number of prompts to time (default 33)")
    ap.add_argument("--model", default=None, help="model type (default: YAML model.type)")
    ap.add_argument("--config", type=Path, default=Path("config/run_config.yml"))
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    bank = args.bank_image or _pick_bank_image(args.data_dir)
    print(f"bank image : {bank}")
    print(f"N prompts  : {args.n}")

    processor, model_cm = _build_processor(args.model, args.data_dir, args.config)
    try:
        backend = processor._backend  # VllmBackend
        model_type = processor.app_config.pipeline.model_type
        vllm_cfg = processor.app_config.get_vllm_config(model_type)
        budget = processor.app_config.get_image_budget("bank_statement")
        max_tiles, min_tiles = int(budget["max_tiles"]), int(budget["min_tiles"])
        print(f"model_type : {model_type}")
        print(f"max_num_seqs (engine concurrency cap): {vllm_cfg.get('max_num_seqs')}")
        print(f"bank tiles : min={min_tiles} max={max_tiles}\n")

        img = Image.open(bank).convert("RGB")
        prompt_tmpl = load_link_prompt("single_receipt_link")

        # N distinct single-image prompts against the SAME statement (the
        # per-statement-batch case: shared prefix + varying receipt suffix).
        convs = []
        for i in range(args.n):
            receipt = ReceiptSummary(
                image_name=f"probe_{i}.png",
                supplier_name=f"Vendor {i}",
                date=date(2024, 1, 1),
                total=100.0 + i,  # vary so each is a distinct request
                document_type="RECEIPT",
            )
            text = build_link_prompt(receipt, prompt=prompt_tmpl, bank_columns=None)
            convs.append(
                backend._build_messages(
                    img, text, image_first=True, max_tiles=max_tiles, min_tiles=min_tiles
                )
            )

        sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0)
        chat_template = backend._chat_template

        # Warm-up: load/compile + prime the shared image prefix so both timings
        # start from the same (warm) cache state — isolating the batching effect.
        print("warming up (load + prime image prefix)...")
        backend.model.chat(
            messages=convs[0], sampling_params=sampling, chat_template=chat_template, use_tqdm=False
        )

        # SERIAL — one chat() per prompt.
        t = time.perf_counter()
        for conv in convs:
            backend.model.chat(
                messages=conv, sampling_params=sampling, chat_template=chat_template, use_tqdm=False
            )
        serial = time.perf_counter() - t

        # BATCHED — one chat() with all N conversations; vLLM runs up to
        # max_num_seqs concurrently.
        t = time.perf_counter()
        backend.model.chat(
            messages=convs, sampling_params=sampling, chat_template=chat_template, use_tqdm=False
        )
        batched = time.perf_counter() - t

        print(
            f"\nRESULT  N={args.n}"
            f"\n  serial : {serial:6.1f}s  ({serial / args.n * 1000:5.0f} ms/call)"
            f"\n  batched: {batched:6.1f}s  ({batched / args.n * 1000:5.0f} ms/call)"
            f"\n  speedup: {serial / batched:.2f}x"
        )
    finally:
        model_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
