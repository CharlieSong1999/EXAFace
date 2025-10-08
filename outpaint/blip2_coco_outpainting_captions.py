#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLIP-2 captioner for COCO "images" entries, tailored for SDXL outpainting.

Now supports:
- Multi-GPU multiprocessing (one worker per GPU by default)
- Grouping by source image id: caption once per source (e.g., 4 crops share one source)

Modes
-----
1) visualize: show specified target image IDs with 10 captions of their *source* image (single-process)
2) work: caption all entries; write JSONL lines including target<->source mapping (multi-process capable)

File resolution
---------------
--source-img-root: directory containing the original/source images
--source-pattern:  filename template (default "{id:012d}.jpg" -> 000000393226.jpg)
Fallbacks: {id}.jpg, {id}.png, and glob search **/*{id}.*
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterable
import os
import re
import math
import itertools
import shutil
import time
import uuid

import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoProcessor, Blip2ForConditionalGeneration

# -----------------------------
# Outpainting-oriented prompts (NO question marks)
# -----------------------------
DECL_PROMPTS = [
    "Brief outpainting caption:",
    "Wider scene, concise:",
    "Surroundings, 8–24 words:",
    "Broader view summary:",
    "Foreground and context:",
    "Backdrop and nearby elements:",
    "Expanded canvas description:",
    "Environmental details, concise:",
    "Scene context beyond edges:",
    "Panoramic context, brief:",
]


def load_coco_images(coco_ann_path: str) -> List[Dict[str, Any]]:
    with open(coco_ann_path, "r") as f:
        data = json.load(f)
    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValueError("Expected 'images' to be a list in the COCO annotation.")
    return images


def build_model_and_processor(model_name: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    hf_processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    return hf_processor, model, device, dtype


def resolve_source_image_path(
    source_id: int,
    root: Path,
    pattern: str = "{id:012d}.jpg",
) -> Optional[Path]:
    """
    Resolve the source image path by id using a pattern, then fallbacks.
    """
    try:
        candidate = root / pattern.format(id=source_id)
        if candidate.exists():
            return candidate
    except Exception:
        pass

    for name in [f"{source_id}.jpg", f"{source_id}.png", f"{source_id:012d}.png"]:
        p = root / name
        if p.exists():
            return p

    for p in root.rglob(f"*{source_id}.*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            return p

    return None


# -----------------------------
# Caption cleaning & filters
# -----------------------------
NUM_PREFIX_RE = re.compile(r"^\s*\d+[\)\.\-_:]\s*")       # strip "05." / "7) " etc.
EMBEDDED_NUM_RE = re.compile(r"(\d{1,2}[\)\.\-_:])")      # strip embedded enumerators
QUOTE_CHARS = '"“”‘’\''

MONTHS = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","sept","oct","nov","dec",
]
WEEKDAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday",
            "mon","tue","tues","wed","thu","thur","thurs","fri","sat","sun"]

DATE_REGEXES = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),             # 03/12/2003, 3-12-03
    re.compile(r"\b(?:19|20)\d{2}\b"),                             # 1999, 2012
    re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4}\b", re.I),
    re.compile(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", re.I),
]

PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")  # Two+ consecutive Capitalized words
ATTRIB_RE = re.compile(r"(?:^|\s)(?:by|©|copyright|photo by|—|-)\s+[A-Za-z].+$", re.I)


def _looks_like_date_or_name(text: str) -> bool:
    low = text.lower()
    if any(m in low for m in MONTHS) or any(d in low for d in WEEKDAYS):
        return True
    for rx in DATE_REGEXES:
        if rx.search(text):
            return True
    if ATTRIB_RE.search(text):
        return True
    # Conservative: if two+ consecutive Capitalized Words appear, treat as a name/location
    if PROPER_NAME_RE.search(text):
        return True
    return False


def _clean_caption(
    text: str,
    *,
    min_words: int,
    max_words: int,
) -> str:
    t = text.strip()

    # Strip leading enumerators like "05." or "7) "
    t = NUM_PREFIX_RE.sub("", t)

    # Remove common leading labels the LM might echo
    for lead in ("Answer:", "Caption:", "Description:", "Context:", "Scene:"):
        if t.lower().startswith(lead.lower()):
            t = t[len(lead):].strip()

    # Remove surrounding quotes (straight or curly)
    t = t.strip(QUOTE_CHARS)

    # Remove trailing bracket/paren annotations like "[14 characters]" or "(12 words)"
    tail_patterns = [
        r"\s*\[[^\]]*\]\s*$",
        r"\s*\([^\)]*\)\s*$",
        r"\s*-\s*\[[^\]]*\]\s*$",
    ]
    changed = True
    while changed:
        changed = False
        for pat in tail_patterns:
            nt = re.sub(pat, "", t)
            if nt != t:
                t = nt
                changed = True
                break

    # Remove small embedded enumerators like "… 05. …"
    t = EMBEDDED_NUM_RE.sub("", t)

    # Normalize repeated punctuation / commas
    t = re.sub(r"\s*,\s*,+", ", ", t)
    t = re.sub(r"\s*-\s*-\s*", " - ", t)
    t = " ".join(t.split())

    # Reject names/dates/attribution/meta
    if _looks_like_date_or_name(t):
        return ""

    # Trim by words
    words = t.split()
    if len(words) < min_words:
        return ""
    if len(words) > max_words:
        t = " ".join(words[:max_words])

    # Drop trailing period
    if t.endswith("."):
        t = t[:-1]

    return t


# -----------------------------
# Generation core
# -----------------------------
def generate_captions_for_image(
    image: Image.Image,
    hf_processor,
    model,
    device: str,
    *,
    bad_words_ids: Optional[List[List[int]]] = None,
    n_captions: int = 10,
    max_new_tokens: int = 60,
    min_new_tokens: int = 12,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    min_words: int = 8,
    max_words: int = 28,
    seed: Optional[int] = None,
    # Hard-stop controls
    max_total_attempts: Optional[int] = None,     # default: 12 * n_captions
    max_fallback_attempts: Optional[int] = None,  # default: 6 * n_captions
    per_image_timeout_sec: Optional[float] = 90.0 # soft wall-clock timeout per image
) -> List[str]:
    """
    Robust caption generation that always returns exactly `n_captions`.

    Guarantees termination via:
      - global attempt cap across both loops,
      - fallback attempt cap,
      - per-image wall-clock timeout,
      - optional relaxation of min_words late in the search.

    If fewer than `n_captions` unique captions are found, fills the remainder by
    randomly repeating existing captions (with replacement).
    """
    import time as _time
    import random

    # Defaults for caps
    if max_total_attempts is None:
        max_total_attempts = n_captions * 12
    if max_fallback_attempts is None:
        max_fallback_attempts = n_captions * 6

    start_t = _time.time()
    captions: List[str] = []
    seen: set[str] = set()
    attempts = 0
    fb_attempts = 0

    def _too_slow() -> bool:
        return per_image_timeout_sec is not None and (_time.time() - start_t) > per_image_timeout_sec

    def _pass_filters(text: str, relax: bool = False) -> str:
        # Reuse global cleaner; optionally relax min_words late in the game
        mw = max(6, min_words) if relax else min_words
        t = _clean_caption(text, min_words=mw, max_words=max_words)
        if not t or t.lower() in {"yes", "no", "maybe"}:
            return ""
        if "?" in text or any(x in text.lower() for x in ["answer", "question", "comments", "why", "how", "what"]):
            return ""
        return t

    def _gen_once(prompt: Optional[str]) -> Optional[str]:
        nonlocal attempts
        if seed is not None:
            torch.manual_seed(seed + attempts + 1)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed + attempts + 1)

        inputs = hf_processor(image, text=prompt, return_tensors="pt").to(device) if prompt \
                 else hf_processor(image, return_tensors="pt").to(device)

        if device == "cuda" and model.dtype == torch.float16:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    inputs[k] = v.half()

        prompt_len = inputs["input_ids"].shape[1] if prompt and "input_ids" in inputs else 0

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=1,
        )
        if bad_words_ids:
            bw = [w for w in bad_words_ids if w]
            if bw:
                gen_kwargs["bad_words_ids"] = bw

        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)

        new_tokens = out_ids[:, prompt_len:] if prompt_len and out_ids.shape[1] > prompt_len else out_ids
        return hf_processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    # -------- Loop 1: prompted sampling (diverse prompts) --------
    while len(captions) < n_captions:
        if attempts >= max_total_attempts or _too_slow():
            break
        prompt = DECL_PROMPTS[attempts % len(DECL_PROMPTS)]
        attempts += 1
        try:
            raw = _gen_once(prompt)
        except Exception:
            continue

        relax = attempts > (max_total_attempts // 2)
        text = _pass_filters(raw or "", relax=relax)
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            captions.append(text)

    # -------- Loop 2: unconditional fallback (if still short) --------
    while len(captions) < n_captions:
        if attempts >= max_total_attempts or fb_attempts >= max_fallback_attempts or _too_slow():
            break
        attempts += 1
        fb_attempts += 1
        try:
            raw = _gen_once(prompt=None)
        except Exception:
            continue
        relax = attempts > (max_total_attempts // 2)
        text = _pass_filters(raw or "", relax=relax)
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            captions.append(text)

    # Warn if we fell short, then pad by random repeats to reach exactly n_captions
    if len(captions) < n_captions:
        reason = []
        if attempts >= max_total_attempts:
            reason.append(f"max_total_attempts={max_total_attempts}")
        if fb_attempts >= max_fallback_attempts:
            reason.append(f"max_fallback_attempts={max_fallback_attempts}")
        if _too_slow():
            reason.append(f"timeout={per_image_timeout_sec}s")
        print(f"[WARN] Had {len(captions)}/{n_captions} unique captions "
              f"({'; '.join(reason) or 'filters rejected many samples'}). Filling by random repeats.")
        if captions:
            while len(captions) < n_captions:
                captions.append(random.choice(captions))
        else:
            captions = ["scene description"] * n_captions

    return captions[:n_captions]


# -----------------------------
# Visualization (single process)
# -----------------------------
def visualize_images_with_captions(
    args: argparse.Namespace,
    images_meta: List[Dict[str, Any]],
    img_root: Path,
    src_root: Path,
    src_pattern: str,
    hf_processor,
    model,
    device: str,
    image_ids_to_show: List[int],
    *,
    bad_words_ids: Optional[List[List[int]]] = None,
):
    id_set = set(image_ids_to_show)
    sel = [im for im in images_meta if int(im.get("id")) in id_set]
    if not sel:
        print("No matching image IDs found in the annotations.")
        return

    for im in sel:
        tgt_id = int(im.get("id"))
        tgt_file = im.get("file_name")
        meta = im.get("meta", {}) or {}

        if args.ignore_source:
            # Directly use the *target* image from --img-root
            tgt_path = img_root / tgt_file
            if not tgt_path.exists():
                print(f"[WARN] Image ID {tgt_id}: target file not found: {tgt_path}")
                continue
            pil_img = Image.open(tgt_path).convert("RGB")
            captions = generate_captions_for_image(
                pil_img, hf_processor, model, device,
                bad_words_ids=bad_words_ids, seed=args.seed
            )

            plt.figure(figsize=(8, 6))
            plt.imshow(pil_img)
            plt.axis("off")
            plt.title(f"[ignore_source] Target ID {tgt_id} [{tgt_path.name}]", fontsize=10)

            print("\n" + "=" * 80)
            print(f"[ignore_source] Target ID: {tgt_id}, file: {tgt_path}")
            for idx, cap in enumerate(captions, 1):
                print(f"{idx:02d}. {cap}")
            print("=" * 80 + "\n")

            box_text = "\n".join([f"{i+1:02d}. {c}" for i, c in enumerate(captions)])
            plt.gcf().text(0.01, -0.05, box_text, ha="left", va="top", fontsize=9, transform=plt.gca().transAxes)
            plt.tight_layout()
            plt.show()
            continue

        # Default behavior: caption the *source* image
        src_id = meta.get("inferred_source_image_id", None)
        if src_id is None:
            print(f"[WARN] Image ID {tgt_id}: no meta.inferred_source_image_id, skipping.")
            continue

        src_path = resolve_source_image_path(int(src_id), src_root, src_pattern)
        if not src_path or not src_path.exists():
            print(f"[WARN] Image ID {tgt_id}: source {src_id} not found under {src_root}")
            continue

        pil_img = Image.open(src_path).convert("RGB")
        captions = generate_captions_for_image(
            pil_img, hf_processor, model, device,
            bad_words_ids=bad_words_ids, seed=args.seed
        )

        plt.figure(figsize=(8, 6))
        plt.imshow(pil_img)
        plt.axis("off")
        plt.title(
            f"Target ID {tgt_id} ({tgt_file}) → Source ID {src_id} [{src_path.name}]",
            fontsize=10,
        )

        print("\n" + "=" * 80)
        print(f"Target ID: {tgt_id}, target file: {tgt_file}")
        print(f"Source ID: {src_id}, source file: {src_path}")
        for idx, cap in enumerate(captions, 1):
            print(f"{idx:02d}. {cap}")
        print("=" * 80 + "\n")

        box_text = "\n".join([f"{i+1:02d}. {c}" for i, c in enumerate(captions)])
        plt.gcf().text(0.01, -0.05, box_text, ha="left", va="top", fontsize=9, transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.show()


# -----------------------------
# Grouping by source id (caption once, reuse for crops)
# -----------------------------
def group_by_source(images_meta: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for im in images_meta:
        meta = im.get("meta", {}) or {}
        sid = meta.get("inferred_source_image_id", None)
        if sid is None:
            # Put under -1 group to still emit a record (missing source)
            sid = -1
        sid = int(sid)
        groups.setdefault(sid, []).append(im)
    return groups


# -----------------------------
# Multiprocessing workers
# -----------------------------
def _worker_init(gpu_id: int, model_name: str, device_str: Optional[str], seed: Optional[int]):
    # Set the process's visible GPU
    if device_str is None:
        # If CUDA is available and gpu_id >= 0, pin to that device id
        if torch.cuda.is_available() and gpu_id >= 0:
            torch.cuda.set_device(gpu_id)
    else:
        # If user passed 'cpu', we won't set CUDA device
        pass

    if seed is not None:
        torch.manual_seed(seed + gpu_id * 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + gpu_id * 1000)

    # Build model on the chosen device
    device = "cuda" if (torch.cuda.is_available() and gpu_id >= 0 and device_str != "cpu") else "cpu"
    hf_processor, model, device, _ = build_model_and_processor(model_name, device=device)
    model.eval()

    # Tokenizer & bad words
    tokenizer = getattr(hf_processor, "tokenizer", None) or getattr(hf_processor, "text_tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    BAD_STRINGS = [
        "Question", "Answer", "question", "answer",
        "context:", "Context:", "comments", "comment",
        "this image", "picture of", "are you", "why", "how", "what",
        "photo", "photograph",
        # months & weekdays to discourage date-y outputs
        *[m.capitalize() for m in MONTHS], *MONTHS,
        *[d.capitalize() for d in WEEKDAYS], *WEEKDAYS,
        "characters", "words", "tokens", "letters",
        # common attribution / metadata patterns
        "by", "©", "copyright"
    ]
    bad_words_ids = None
    if tokenizer:
        bad_words_ids = [tokenizer(b, add_special_tokens=False).input_ids for b in BAD_STRINGS]
        qmark_ids = tokenizer("?", add_special_tokens=False).input_ids
        if qmark_ids:
            bad_words_ids.append(qmark_ids)
        bad_words_ids = [w for w in bad_words_ids if w]  # drop empties

    # Save in global module vars for the worker
    globals()["_W_GPU_ID"] = gpu_id
    globals()["_W_DEVICE"] = device
    globals()["_W_HF_PROCESSOR"] = hf_processor
    globals()["_W_MODEL"] = model
    globals()["_W_BAD_WORDS_IDS"] = bad_words_ids
    globals()["_W_SEED"] = seed


def _worker_task(
    args: Tuple[int, str, str, str, List[Tuple[int, str, List[Dict[str, Any]]]], str, int],
    progress=None,
    pbar=None,
):
    """
    Worker entry: receives a shard of groups.

    args:
      gpu_id, source_root, source_pattern, out_dir, group_list, model_name, rank
    returns: path to .part file
    """
    (gpu_id, source_root, source_pattern, out_dir,
     group_list, model_name, rank) = args

    # Ensure model/processor are initialized in this process
    hf_processor = globals()["_W_HF_PROCESSOR"]
    model = globals()["_W_MODEL"]
    device = globals()["_W_DEVICE"]
    bad_words_ids = globals()["_W_BAD_WORDS_IDS"]
    seed = globals()["_W_SEED"]

    src_root = Path(source_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"captions.part{rank}.jsonl"

    with part_path.open("w", encoding="utf-8") as f:
        for (src_id, src_path_str, targets) in group_list:
            try:
                # caption per source (if src_id == -1 or missing path => emit error lines for all targets)
                if src_id == -1 or not src_path_str:
                    err = "no inferred_source_image_id in meta" if src_id == -1 else f"source not found under {source_root}"
                    for im in targets:
                        rec = {
                            "target_image_id": int(im.get("id")),
                            "target_file_name": im.get("file_name"),
                            **({"source_image_id": int(src_id)} if src_id != -1 else {}),
                            "captions": [],
                            "error": err,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                else:
                    src_path = Path(src_path_str)
                    if not src_path.exists():
                        for im in targets:
                            rec = {
                                "target_image_id": int(im.get("id")),
                                "target_file_name": im.get("file_name"),
                                "source_image_id": int(src_id),
                                "captions": [],
                                "error": f"source not found under {source_root}",
                            }
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    else:
                        pil_img = Image.open(src_path).convert("RGB")
                        captions = generate_captions_for_image(
                            pil_img, hf_processor, model, device,
                            bad_words_ids=bad_words_ids, seed=seed
                        )
                        for im in targets:
                            rec = {
                                "target_image_id": int(im.get("id")),
                                "target_file_name": im.get("file_name"),
                                "source_image_id": int(src_id),
                                "source_file_path": str(src_path),
                                "captions": captions,
                            }
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            except Exception as e:
                # robust error line per target
                for im in targets:
                    rec = {
                        "target_image_id": int(im.get("id")),
                        "target_file_name": im.get("file_name"),
                        **({"source_image_id": int(src_id)} if src_id != -1 else {}),
                        **({"source_file_path": src_path_str} if src_path_str else {}),
                        "captions": [],
                        "error": repr(e),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # ---- progress: exactly once per source group ----
            if progress is not None:
                with progress.get_lock():
                    progress.value += 1
            if pbar is not None:
                pbar.update(1)

    return str(part_path)


# -----------------------------
# Work mode: multi-process orchestration
# -----------------------------
def run_work_mode_mp(
    args: argparse.Namespace,
    images_meta: List[Dict[str, Any]],
    src_root: Path,
    src_pattern: str,
) -> None:
    import multiprocessing as mp
    import threading

    # --- Build list of "groups" ---
    group_items: List[Tuple[int, str, List[Dict[str, Any]]]] = []

    if args.ignore_source:
        # One group per *image*, path from --img-root / file_name
        img_root = Path(args.img_root)
        for im in images_meta:
            im_id = int(im.get("id"))
            f = im.get("file_name")
            p = img_root / f if f else None
            group_items.append((im_id, str(p) if p and p.exists() else "", [im]))
        root_for_errors = img_root
        desc = "Captioning (images)"
        unit = "img"
    else:
        # Original behavior: group by source id, caption once per source
        groups = group_by_source(images_meta)
        for src_id, targets in groups.items():
            if src_id == -1:
                group_items.append((src_id, "", targets))
            else:
                p = resolve_source_image_path(src_id, src_root, src_pattern)
                group_items.append((src_id, str(p) if p is not None else "", targets))
        root_for_errors = src_root
        desc = "Captioning (source groups)"
        unit = "group"

    total_groups = len(group_items)

    # GPU assignment
    if args.gpus:
        gspec = args.gpus.strip()
        if gspec == "-1":
            gpu_ids = [-1]  # CPU only
        else:
            gpu_ids = [int(x) for x in gspec.split(",") if x.strip() != ""]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [-1]

    num_workers = args.num_workers if args.num_workers is not None else max(1, len(gpu_ids))
    workers_gpu_map = [gpu_ids[i % len(gpu_ids)] for i in range(num_workers)]

    # Round-robin assign groups to workers
    shards: List[List[Tuple[int, str, List[Dict[str, Any]]]]] = [[] for _ in range(num_workers)]
    for idx, item in enumerate(group_items):
        shards[idx % num_workers].append(item)

    # Prepare tmp dir for .part files
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / f".parts_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Shared counter for progress
    ctx = mp.get_context("spawn")
    progress = ctx.Value('i', 0)

    # tqdm progress monitor (thread)
    from tqdm import tqdm as _tqdm
    pbar = _tqdm(total=total_groups, desc=desc, unit=unit)
    stop_evt = threading.Event()

    def _monitor():
        last = 0
        while not stop_evt.is_set():
            try:
                val = progress.value
                if val > last:
                    pbar.update(val - last)
                    last = val
                time.sleep(0.2)
            except Exception:
                break

    mon_thr = threading.Thread(target=_monitor, daemon=True)
    mon_thr.start()

    processes = []
    ret_paths = []

    try:
        # --- Run rank 0 shard in MAIN process (uses its assigned GPU/CPU) ---
        if num_workers >= 1:
            rank0_gpu = workers_gpu_map[0]
            args_tuple0 = (rank0_gpu, str(root_for_errors), src_pattern, str(tmp_dir), shards[0], args.model, 0)

            _worker_init(rank0_gpu, args.model, args.device, args.seed)
            _worker_task(args_tuple0, progress=progress, pbar=pbar)

        # --- Spawn the rest (ranks 1..N-1) ---
        for rank in range(1, num_workers):
            gpu_id = workers_gpu_map[rank]
            shard = shards[rank]
            args_tuple = (gpu_id, str(root_for_errors), src_pattern, str(tmp_dir), shard, args.model, rank)
            p = ctx.Process(
                target=_worker_entrypoint,
                args=(args_tuple, args.device, args.seed, progress),
                name=f"blip2_caption_worker_{rank}"
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"[WARN] Worker {p.name} exited with code {p.exitcode}")

        final_val = progress.value
        if final_val < total_groups:
            pbar.update(total_groups - final_val)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt: terminating workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join()
        raise
    finally:
        stop_evt.set()
        mon_thr.join(timeout=1.0)
        pbar.close()

    # Collect .part files
    for rank in range(num_workers):
        part_file = tmp_dir / f"captions.part{rank}.jsonl"
        if part_file.exists():
            ret_paths.append(str(part_file))
        else:
            print(f"[WARN] Missing part file for worker {rank}: {part_file}")

    # Merge into final JSONL
    with out_path.open("w", encoding="utf-8") as out_f:
        for part in ret_paths:
            with open(part, "r", encoding="utf-8") as pf:
                shutil.copyfileobj(pf, out_f)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\nDone. Wrote JSONL to: {out_path}")


def _worker_entrypoint(args_tuple, device_str: Optional[str], seed: Optional[int], progress):
    """
    Per-process entrypoint; constructs the model on the assigned GPU and runs shard.
    """
    (gpu_id, source_root, source_pattern, out_dir,
     shard, model_name, rank) = args_tuple

    _worker_init(gpu_id, model_name, device_str, seed)
    _worker_task(args_tuple, progress=progress, pbar=None)


# -----------------------------
# Work mode: single-process (fallback)
# -----------------------------
def run_work_mode_single(
    args: argparse.Namespace,
    images_meta: List[Dict[str, Any]],
    hf_processor,
    model,
    device: str,
    src_root: Path,
    src_pattern: str,
    bad_words_ids: Optional[List[List[int]]],
) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.ignore_source:
        img_root = Path(args.img_root)
        with out_path.open("w", encoding="utf-8") as f:
            for im in tqdm(images_meta, desc="Captioning (images)", unit="img"):
                im_id = int(im.get("id"))
                f_name = im.get("file_name")
                tgt_path = img_root / f_name if f_name else None

                if not tgt_path or not tgt_path.exists():
                    rec = {
                        "target_image_id": im_id,
                        "target_file_name": f_name,
                        "captions": [],
                        "error": f"target not found under {str(img_root)}",
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue

                try:
                    pil_img = Image.open(tgt_path).convert("RGB")
                    captions = generate_captions_for_image(
                        pil_img, hf_processor, model, device,
                        bad_words_ids=bad_words_ids, seed=args.seed
                    )
                    rec = {
                        "target_image_id": im_id,
                        "target_file_name": f_name,
                        "source_image_id": im_id,           # treat self as source for consistency
                        "source_file_path": str(tgt_path),  # path used as input
                        "captions": captions,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception as e:
                    rec = {
                        "target_image_id": im_id,
                        "target_file_name": f_name,
                        "source_image_id": im_id,
                        "source_file_path": str(tgt_path),
                        "captions": [],
                        "error": repr(e),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nDone. Wrote JSONL to: {out_path}")
        return

    # ---- default behavior: group by source ----
    groups = group_by_source(images_meta)
    with out_path.open("w", encoding="utf-8") as f:
        for src_id, targets in tqdm(groups.items(), desc="Captioning (source groups)", unit="group"):
            if src_id == -1:
                for im in targets:
                    rec = {
                        "target_image_id": int(im.get("id")),
                        "target_file_name": im.get("file_name"),
                        "captions": [],
                        "error": "no inferred_source_image_id in meta",
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            src_path = resolve_source_image_path(src_id, src_root, src_pattern)
            if not src_path or not src_path.exists():
                for im in targets:
                    rec = {
                        "target_image_id": int(im.get("id")),
                        "target_file_name": im.get("file_name"),
                        "source_image_id": int(src_id),
                        "captions": [],
                        "error": f"source not found under {str(src_root)}",
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            try:
                pil_img = Image.open(src_path).convert("RGB")
                captions = generate_captions_for_image(
                    pil_img, hf_processor, model, device,
                    bad_words_ids=bad_words_ids, seed=args.seed
                )
                for im in targets:
                    rec = {
                        "target_image_id": int(im.get("id")),
                        "target_file_name": im.get("file_name"),
                        "source_image_id": int(src_id),
                        "source_file_path": str(src_path),
                        "captions": captions,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                for im in targets:
                    rec = {
                        "target_image_id": int(im.get("id")),
                        "target_file_name": im.get("file_name"),
                        "source_image_id": int(src_id),
                        "source_file_path": str(src_path),
                        "captions": [],
                        "error": repr(e),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. Wrote JSONL to: {out_path}")


# -----------------------------
# CLI & Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="BLIP-2 COCO captioner for SDXL outpainting (source-image aware, multi-GPU)")
    p.add_argument("--ann", type=str, required=True, help="Path to COCO annotation JSON")
    p.add_argument("--img-root", type=str, required=True, help="Root folder that contains *target* images (kept for reference)")
    p.add_argument("--mode", type=str, choices=["visualize", "work"], required=True)
    p.add_argument("--image-ids", type=str, default="", help="Comma-separated target image IDs (visualize mode)")
    p.add_argument("--out", type=str, default="captions.jsonl", help="Output JSONL path (work mode)")
    p.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b",
                   help="HF model id, e.g., Salesforce/blip2-opt-2.7b or Salesforce/blip2-flan-t5-xl")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if not set)")
    p.add_argument("--source-img-root", type=str, default=None,
                   help="Folder with original/source images (default: --img-root)")
    p.add_argument("--source-pattern", type=str, default="{id:012d}.jpg",
                   help='Filename template for source images, e.g. "{id:012d}.jpg" or "{id}.jpg"')
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # NEW: ignore source flag (supports both spellings)
    p.add_argument("--ignore-source", "--ignore_source", dest="ignore_source",
                   action="store_true",
                   help="Caption every image in --ann directly from --img-root (do not group by inferred_source_image_id)")

    # Multiprocessing controls (work mode)
    p.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids to use (default: all visible). Use -1 for CPU.")
    p.add_argument("--num-workers", type=int, default=None, help="Number of worker processes (default: #GPUs or 1).")

    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    img_root = Path(args.img_root)
    src_root = Path(args.source_img_root) if args.source_img_root else img_root

    images_meta = load_coco_images(args.ann)

    # Visualize = single process
    if args.mode == "visualize":
        hf_processor, model, device, _ = build_model_and_processor(args.model, args.device)
        model.eval()

        tokenizer = getattr(hf_processor, "tokenizer", None) or getattr(hf_processor, "text_tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        BAD_STRINGS = [
            "Question", "Answer", "question", "answer",
            "context:", "Context:", "comments", "comment",
            "this image", "picture of", "are you", "why", "how", "what",
            "photo", "photograph",
            *[m.capitalize() for m in MONTHS], *MONTHS,
            *[d.capitalize() for d in WEEKDAYS], *WEEKDAYS,
            "characters", "words", "tokens", "letters",
        ]
        bad_words_ids = None
        if tokenizer:
            bad_words_ids = [tokenizer(b, add_special_tokens=False).input_ids for b in BAD_STRINGS]
            qmark_ids = tokenizer("?", add_special_tokens=False).input_ids
            if qmark_ids:
                bad_words_ids.append(qmark_ids)
            bad_words_ids = [w for w in bad_words_ids if w]

        if not args.image_ids:
            raise ValueError("Please provide --image-ids for visualize mode (e.g., '1,5,42').")
        image_ids = [int(x.strip()) for x in args.image_ids.split(",") if x.strip()]
        visualize_images_with_captions(
            args, images_meta, img_root, src_root, args.source_pattern,
            hf_processor, model, device, image_ids,
            bad_words_ids=bad_words_ids
        )
        return

    # Work mode
    # If only one worker (CPU or single GPU), keep it simple: single-process path
    if args.num_workers is None and (not torch.cuda.is_available() or torch.cuda.device_count() <= 1) and not args.gpus:
        hf_processor, model, device, _ = build_model_and_processor(args.model, args.device)
        model.eval()

        tokenizer = getattr(hf_processor, "tokenizer", None) or getattr(hf_processor, "text_tokenizer", None)
        bad_words_ids = None
        if tokenizer:
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            BAD_STRINGS = [
                "Question", "Answer", "question", "answer",
                "context:", "Context:", "comments", "comment",
                "this image", "picture of", "are you", "why", "how", "what",
                "photo", "photograph",
                *[m.capitalize() for m in MONTHS], *MONTHS,
                *[d.capitalize() for d in WEEKDAYS], *WEEKDAYS,
                "characters", "words", "tokens", "letters",
            ]
            bad_words_ids = [tokenizer(b, add_special_tokens=False).input_ids for b in BAD_STRINGS]
            qmark_ids = tokenizer("?", add_special_tokens=False).input_ids
            if qmark_ids:
                bad_words_ids.append(qmark_ids)
            bad_words_ids = [w for w in bad_words_ids if w]

        run_work_mode_single(
            args, images_meta, hf_processor, model, device, src_root, args.source_pattern, bad_words_ids
        )
    else:
        run_work_mode_mp(args, images_meta, src_root, args.source_pattern)


if __name__ == "__main__":
    main()
