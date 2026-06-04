#!/usr/bin/env python3
"""Self-contained smoke test: is InternVL3.5 emitting <think> reasoning?

Loads the served model once and runs two chat calls so you can tell, in ~30s,
WHICH fix you need before committing to a multi-hour classify+extract re-run:

  1. "default"        -> no system message (what the bare template/model does)
  2. "no-think system"-> our benign non-thinking system message (Change 1)

Verdict:
  * default thinks, no-think-system does NOT  -> the system message is enough.
  * BOTH still think                          -> the template forces thinking;
                                                 use the model.chat_template knob.
  * neither thinks                            -> nothing to do.

Pure vLLM + PIL — no project imports, so it works even on a bare box.

Usage (on the 2xL4 cluster):
    python scripts/check_thinking.py \
        --model /home/jovyan/nfs_share/models/InternVL3_5-8B --tp 2

    # To reproduce the real detection call, add an image:
    python scripts/check_thinking.py --model <path> --tp 2 \
        --image ../evaluation_data/synthetic_transaction_linking/CASE007_anz_standard.png
"""

import argparse
import base64
import io
import sys

NO_THINK_SYSTEM = (
    "You are a precise document-analysis assistant. Answer directly and "
    "concisely in exactly the format requested. Do not show your reasoning."
)

# A detection-style prompt (image case) and a trivial one (text-only case).
IMAGE_PROMPT = (
    "Answer in EXACTLY this format, three lines, nothing else:\n"
    "1. COLUMNS: <headers separated by ' | ', or NONE>\n"
    "2. PAID: <YES or NO>\n"
    "3. ROWS: <number>"
)
TEXT_PROMPT = "Reply with the single word OK."


def _data_uri(path: str) -> str:
    from PIL import Image

    with Image.open(path) as raw:
        buf = io.BytesIO()
        raw.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _user_content(image: str | None) -> object:
    prompt = IMAGE_PROMPT if image else TEXT_PROMPT
    if not image:
        return prompt
    # text-first ordering, matching the detection path
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _data_uri(image)}},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Path to the InternVL3.5 model directory")
    ap.add_argument("--tp", type=int, default=2, help="tensor_parallel_size (default 2 for 2xL4)")
    ap.add_argument("--image", default=None, help="Optional image to reproduce the detection call")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm_kwargs: dict[str, object] = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "enforce_eager": True,
    }
    if args.image:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 1}

    print(f"Loading {args.model} (tp={args.tp}, image={'yes' if args.image else 'no'}) ...")
    engine = LLM(**llm_kwargs)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0)
    content = _user_content(args.image)

    cases = {
        "default (no system message)": [{"role": "user", "content": content}],
        "no-think system message": [
            {"role": "system", "content": NO_THINK_SYSTEM},
            {"role": "user", "content": content},
        ],
    }

    thinks: dict[str, bool] = {}
    for name, messages in cases.items():
        out = engine.chat(messages=messages, sampling_params=sampling, use_tqdm=False)
        text = out[0].outputs[0].text
        has_think = "<think>" in text.lower()
        thinks[name] = has_think
        print("\n" + "=" * 70)
        print(f"CASE: {name}   ->   <think> present: {has_think}")
        print("-" * 70)
        print(text[:600])

    default_thinks = thinks["default (no system message)"]
    system_thinks = thinks["no-think system message"]
    print("\n" + "#" * 70)
    if not default_thinks and not system_thinks:
        print("VERDICT: No <think> in either case — thinking is already OFF. Nothing to do.")
    elif default_thinks and not system_thinks:
        print("VERDICT: The no-think SYSTEM MESSAGE suppresses thinking (Change 1 is enough).")
        print("         Re-run classify+extract; no chat_template override needed.")
    else:
        print("VERDICT: The TEMPLATE forces thinking (system message did not stop it).")
        print("         Use the knob: set model.chat_template to a non-thinking *.jinja")
        print("         in config/run_config.yml, then re-run.")
    print("#" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
