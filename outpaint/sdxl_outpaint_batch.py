#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDXL Outpainting (multi-GPU, tune mode, working-size, multi-guidance).

Key features:
- Multi-GPU: one worker process per GPU, indices assigned round-robin.
- Tune mode: run on a tiny subset (--mode tune --tune_indices / --tune_file).
- Working-size preprocessing: resize to --work_long_side before diffusion.
- Robust masks: center kept, outside painted; optional blur; strict L/0-255.
- Multi-guidance: pass CSV to --guidance (alias --guidence).
- Diagnostics: filename checks, size logs, optional debug saves.

Usage (4 GPUs):
  python sdxl_outpaint_batch.py \
    --gpus 0,1,2,3 \
    --val_img_folder /path/to/val_imgs \
    --captions /path/to/captions.jsonl \
    --out_dir /path/to/out \
    --mode tune --tune_indices "1-8" \
    --guidance 2.5,5.0,7.5 \
    --steps 25 --strength 1.0 --mask_blur 24 \
    --work_long_side 1024 --paste_original_center --save_inputs
"""

# --- Harden against HPC user-site contamination ---
import os, sys, site
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# Drop user site paths like ~/.local/lib/pythonX.Y/site-packages
try:
    user_site = site.getusersitepackages()
    sys.path = [p for p in sys.path if p != user_site and not p.startswith(os.path.expanduser("~/.local/"))]
except Exception:
    pass
# -----------------------------------------------

import argparse, json, logging, math, os, re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
from multiprocessing import Process, set_start_method

# --------------------------- Logging ---------------------------

def setup_logger(verbosity: int = 1, prefix: str = ""):
    level = logging.INFO if verbosity >= 1 else logging.WARNING
    fmt = "%(asctime)s | %(levelname)s | " + (f"[{prefix}] " if prefix else "") + "%(message)s"
    logging.basicConfig(format=fmt, level=level)
    
# helper

import atexit, signal, time, os
from multiprocessing import Process

_CHILD_PROCS: list[Process] = []

def _terminate_children(reason: str, kill_after: float = 10.0):
    # Gracefully terminate then hard-kill if needed
    alive = [p for p in _CHILD_PROCS if p.is_alive()]
    if not alive:
        return
    try:
        import logging
        logging.warning(f"Shutting down {len(alive)} workers ({reason})...")
    except Exception:
        pass
    for p in alive:
        try: p.terminate()
        except Exception: pass
    # wait a bit
    deadline = time.time() + kill_after
    for p in alive:
        try: p.join(timeout=max(0.0, deadline - time.time()))
        except Exception: pass
    # force kill stubborn ones
    for p in alive:
        if p.is_alive():
            try: os.kill(p.pid, signal.SIGKILL)
            except Exception: pass

def _signal_handler(signum, frame):
    names = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM", signal.SIGQUIT: "SIGQUIT"}
    _terminate_children(names.get(signum, f"signal {signum}"))
    # Exit fast; parent will die, daemonic children (if any) will be reaped too
    os._exit(1)  # nosec

def _install_signal_handlers():
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
        signal.signal(sig, _signal_handler)
    atexit.register(lambda: _terminate_children("atexit"))

# --------------------------- Parse helpers ---------------------------

def parse_shard(spec: str) -> Optional[tuple[int, int]]:
    """
    Parse 'R/N' into (rank, world). Returns None if empty.
    Ensures 0 <= R < N and N >= 1.
    """
    if not spec or not spec.strip():
        return None
    s = spec.strip()
    if "/" not in s:
        raise ValueError(f"Invalid --shard '{s}', expected R/N (e.g., 1/4)")
    r, n = s.split("/", 1)
    r, n = r.strip(), n.strip()
    if not r.isdigit() or not n.isdigit():
        raise ValueError(f"Invalid --shard '{s}', expected integers R/N")
    rank, world = int(r), int(n)
    if world <= 0:
        raise ValueError("--shard world size N must be >= 1")
    if not (0 <= rank < world):
        raise ValueError(f"--shard rank must satisfy 0 <= R < N (got {rank}/{world})")
    return rank, world

def parse_indices_spec(spec: str) -> Set[int]:
    out: Set[int] = set()
    if not spec:
        return out
    for tok in re.split(r"[,\s]+", spec.strip()):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a, b = a.strip(), b.strip()
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Invalid range '{tok}'")
            lo, hi = int(a), int(b)
            if hi < lo: lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            if not tok.isdigit():
                raise ValueError(f"Invalid index '{tok}'")
            out.add(int(tok))
    return out

def parse_indices_file(path: Path) -> Set[int]:
    out: Set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line: continue
            try:
                out |= parse_indices_spec(line)
            except ValueError as e:
                logging.warning(f"Skipping tune-file line {line_no}: {e}")
    return out

def parse_float_list(spec: str) -> List[float]:
    if not spec: return []
    vals: List[float] = []
    for tok in re.split(r"[,\s]+", spec.strip()):
        if not tok: continue
        vals.append(float(tok))
    return vals

def parse_gpus(spec: str, want_cuda: bool) -> List[int]:
    if not want_cuda or not torch.cuda.is_available():
        return []
    if not spec.strip():
        return list(range(torch.cuda.device_count()))
    ids = []
    for tok in re.split(r"[,\s]+", spec.strip()):
        if not tok: continue
        ids.append(int(tok))
    # sanity: only keep those that exist
    max_id = torch.cuda.device_count() - 1
    valid = [g for g in ids if 0 <= g <= max_id]
    if not valid:
        raise ValueError(f"No valid GPU ids from '{spec}'. Visible: 0..{max_id}")
    return valid

# --------------------------- Captions ---------------------------

def load_captions_jsonl(jsonl_path: Path) -> Dict[int, List[str]]:
    id_to_captions: Dict[int, List[str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping JSONL line {line_no}: {e}"); continue
            if "target_image_id" not in obj or "captions" not in obj:
                logging.warning(f"Skipping line {line_no}: missing 'target_image_id' or 'captions'"); continue
            tid = int(obj["target_image_id"])
            caps = obj["captions"]
            if not isinstance(caps, list) or len(caps) == 0:
                logging.warning(f"Skipping line {line_no}: 'captions' must be a non-empty list"); continue
            id_to_captions[tid] = [str(c).strip() for c in caps][:10]
    return id_to_captions

# --------------------------- Canvas & Mask ---------------------------

def make_extended_canvas_and_masks(img: Image.Image):
    img = img.convert("RGB")
    w, h = img.size
    ext_w, ext_h = 3 * w, 3 * h
    I_ext = Image.new("RGB", (ext_w, ext_h), (0, 0, 0))
    I_ext.paste(img, (w, h))
    M_ext_center_one = Image.new("L", (ext_w, ext_h), 0)
    M_ext_center_one.paste(Image.new("L", (w, h), 255), (w, h))
    inpaint_mask = ImageOps.invert(M_ext_center_one)  # white=paint
    center_bbox = M_ext_center_one.getbbox()
    return I_ext, M_ext_center_one, inpaint_mask, center_bbox, (w, h)

def _normalize_mask_L(mask: Image.Image) -> Image.Image:
    return mask if mask.mode == "L" else mask.convert("L")

def force_multiples_of_8(im: Image.Image, mask: Image.Image):
    w, h = im.size
    new_w = (w + 7) // 8 * 8
    new_h = (h + 7) // 8 * 8
    if new_w == w and new_h == h:
        return im, mask, (0, 0)
    pad_w, pad_h = new_w - w, new_h - h
    im_padded = Image.new("RGB", (new_w, new_h), (0, 0, 0)); im_padded.paste(im, (0, 0))
    m_padded = Image.new("L", (new_w, new_h), 0); m_padded.paste(mask, (0, 0))
    return im_padded, m_padded, (pad_w, pad_h)

def resize_to_working(im: Image.Image, mask: Image.Image, long_side: int, multiple: int = 8):
    assert long_side > 0
    w, h = im.size
    scale = long_side / float(max(w, h))
    wrk_w = max(1, int(round(w * scale)))
    wrk_h = max(1, int(round(h * scale)))
    # snap to multiple
    wrk_w = (wrk_w + multiple - 1) // multiple * multiple
    wrk_h = (wrk_h + multiple - 1) // multiple * multiple
    im_wrk = im.resize((wrk_w, wrk_h), Image.LANCZOS)
    mask_wrk = _normalize_mask_L(mask).resize((wrk_w, wrk_h), Image.BILINEAR)
    return im_wrk, mask_wrk, (wrk_w, wrk_h)

# --------------------------- Pipeline ---------------------------

def build_pipeline(model_id, vae_id, device_str, use_fp16, cpu_offload):
    try:
        import torch
        from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
    except Exception as e:
        msg = (
            "Failed to import Diffusers/Torch. On HPCs this is usually a NumPy/ABI issue.\n"
            "Recommended: run via Apptainer/Singularity (see README) or conda-pack.\n"
            f"Original import error: {e}"
        )
        raise RuntimeError(msg)

    dtype = torch.float16 if use_fp16 and device_str.startswith("cuda") else torch.float32
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype) if vae_id else None

    candidates = [model_id]
    if model_id != "diffusers/stable-diffusion-xl-1.0-inpainting-0.1":
        candidates.append("diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    candidates.append("OzzyGT/RealVisXL_V4.0_inpainting")

    last_err = None; pipe = None
    for mid in candidates:
        try:
            logging.info(f"Loading SDXL inpaint model: {mid} on {device_str}")
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                mid, torch_dtype=dtype, vae=vae, variant=("fp16" if dtype==torch.float16 else None)
            )
            break
        except Exception as e:
            logging.warning(f"Failed to load {mid}: {e}"); last_err = e
    if pipe is None:
        raise RuntimeError(f"Could not load any SDXL inpaint model from {candidates}") from last_err

    # Safe feature toggles (xformers optional)
    pipe = pipe.to(device_str if device_str != "cuda" else "cuda")
    if device_str.startswith("cuda") and torch.cuda.is_available():
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass   # ok if missing
        try: pipe.enable_attention_slicing(); pipe.enable_vae_slicing()
        except Exception: pass
        if cpu_offload:
            try: pipe.enable_model_cpu_offload()
            except Exception: pass
    return pipe

# --------------------------- SDXL Outpaint ---------------------------

def outpaint_one(
    pipe: StableDiffusionXLInpaintPipeline,
    I_wrk: Image.Image,
    mask_wrk: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    strength: float,
    blur: int,
    seed: Optional[int],
    save_mask_path: Optional[Path] = None,
) -> Image.Image:
    mask_wrk = _normalize_mask_L(mask_wrk)
    mask_used = pipe.mask_processor.blur(mask_wrk, blur_factor=blur) if (blur and blur > 0) else mask_wrk
    if save_mask_path is not None:
        try: mask_used.save(save_mask_path)
        except Exception as e: logging.warning(f"Failed to save debug mask at {save_mask_path}: {e}")
    generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=I_wrk,
        mask_image=mask_used,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
    ).images[0]
    return image

# --------------------------- File collection + diagnostics ---------------------------

def collect_indexed_images(folder: Path, required_len: int = 12,
                           allowed_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")):
    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in allowed_exts}
    all_candidates = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    accepted, rejected = [], []
    for p in all_candidates:
        stem = p.stem.strip(); reasons=[]
        if not stem.isdigit(): reasons.append("stem-not-numeric")
        if len(stem) != required_len: reasons.append(f"stem-len-{len(stem)}!= {required_len}")
        (accepted if not reasons else rejected).append((p if not reasons else (p, ", ".join(reasons))))
    accepted_paths = sorted([p for p in accepted])
    return accepted_paths, all_candidates, rejected

def log_no_match_diagnostics(folder: Path, all_candidates: List[Path], rejected, required_len: int):
    logging.error(f"No files matched %0{required_len}d in {folder}")
    all_files = [p for p in folder.iterdir() if p.is_file()]
    ext_counts = Counter(p.suffix.lower() for p in all_files)
    logging.info(f"Folder has {len(all_files)} files by ext: {dict(ext_counts)}")
    if all_candidates:
        for p, reason in rejected[:20]:
            logging.info(f"Rejected: {p.name} -> {reason}")
    else:
        for name in [p.name for p in all_files[:20]]:
            logging.info(f"  - {name}")

# --------------------------- Worker args ---------------------------

@dataclass
class WorkerArgs:
    gpu_id: int
    device_str: str
    indices: List[int]
    out_dir: str
    val_img_folder: str
    captions_path: str
    # core params
    model_id: str
    vae_id: Optional[str]
    steps: int
    strength: float
    mask_blur: int
    negative_prompt: str
    paste_original_center: bool
    save_inputs: bool
    base_seed: Optional[int]
    verbose: int
    # working-size
    work_long_side: int
    work_multiple: int
    # multi-guidance (list)
    guidance_values: List[float]

# --------------------------- Worker loop ---------------------------

def worker_main(wargs: WorkerArgs):
    import signal, sys
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    setup_logger(wargs.verbose, prefix=f"GPU{wargs.gpu_id}")
    # Load captions once per worker
    id_to_caps = load_captions_jsonl(Path(wargs.captions_path))
    # Map index -> path
    folder = Path(wargs.val_img_folder)
    index_to_path: Dict[int, Path] = {}
    for idx in wargs.indices:
        p = folder / f"{idx:012d}.jpg"
        if not p.exists():
            logging.warning(f"[{idx:012d}] missing file {p.name}, skipping")
        else:
            index_to_path[idx] = p
    if not index_to_path:
        logging.info("No files to process on this worker."); return

    # Build pipeline on this GPU
    use_fp16 = wargs.device_str.startswith("cuda")
    pipe = build_pipeline(
        model_id=wargs.model_id,
        vae_id=wargs.vae_id if wargs.vae_id else None,
        device_str=wargs.device_str,
        use_fp16=use_fp16,
        cpu_offload=False,  # per-GPU process, usually no offload needed
    )

    for index in tqdm(sorted(index_to_path.keys()), desc=f"GPU{wargs.gpu_id}"):
        img_path = index_to_path[index]
        if index not in id_to_caps:
            logging.warning(f"[{index:012d}] no captions, skipping"); continue
        caps = id_to_caps[index][:10] if len(id_to_caps[index])>0 else [""]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.warning(f"[{index:012d}] open failed: {e}"); continue

        I_ext, M_center1, mask_out, center_bbox, (w, h) = make_extended_canvas_and_masks(img)
        logging.info(f"[{index:012d}] ext={I_ext.size}, center_bbox={center_bbox}, orig={w}x{h}")

        # Working-size preprocess
        used_working = False
        if wargs.work_long_side and wargs.work_long_side > 0:
            I_wrk, mask_wrk, (wrk_w, wrk_h) = resize_to_working(I_ext, mask_out, wargs.work_long_side, wargs.work_multiple)
            logging.info(f"[{index:012d}] working={wrk_w}x{wrk_h}")
            used_working = True
        else:
            I_wrk, mask_wrk, (pad_w, pad_h) = force_multiples_of_8(I_ext, mask_out)
            if pad_w or pad_h:
                logging.info(f"[{index:012d}] padded +({pad_w},{pad_h}) -> {I_wrk.size}")

        out_subdir = Path(wargs.out_dir) / f"{index:012d}"
        out_subdir.mkdir(parents=True, exist_ok=True)

        if wargs.save_inputs:
            try:
                I_ext.save(out_subdir / f"{index:012d}_I_ext.jpg")
                M_center1.save(out_subdir / f"{index:012d}_M_ext_center1.png")
                mask_out.save(out_subdir / f"{index:012d}_mask_outpaint.png")
                if used_working:
                    I_wrk.save(out_subdir / f"{index:012d}_I_working.jpg")
                    mask_wrk.save(out_subdir / f"{index:012d}_mask_working.png")
            except Exception as e:
                logging.warning(f"[{index:012d}] failed to save debug inputs: {e}")

        for j, prompt in enumerate(caps):
            seed = None if wargs.base_seed is None else (wargs.base_seed + index * 1000 + j)
            for g in wargs.guidance_values:
                try:
                    out_wrk = outpaint_one(
                        pipe=pipe,
                        I_wrk=I_wrk, mask_wrk=mask_wrk,
                        prompt=prompt, negative_prompt=wargs.negative_prompt,
                        steps=wargs.steps, guidance=g, strength=wargs.strength,
                        blur=wargs.mask_blur, seed=seed,
                        save_mask_path=(out_subdir / f"{index:012d}_mask_used_cap{j:02d}_g{g:04.2f}.png")
                            if wargs.save_inputs else None,
                    )
                except Exception as e:
                    logging.warning(f"[{index:012d}] cap#{j} g={g}: {e}")
                    continue

                # Restore to original extended size
                if used_working:
                    out_img = out_wrk.resize(I_ext.size, Image.LANCZOS)
                else:
                    if I_wrk.size != I_ext.size:
                        out_img = out_wrk.crop((0, 0, I_ext.width, I_ext.height))
                    else:
                        out_img = out_wrk

                # Paste original center exactly at mask bbox
                if wargs.paste_original_center and center_bbox is not None:
                    left, top, right, bottom = center_bbox
                    cw, ch = right - left, bottom - top
                    if (cw, ch) != (w, h):
                        logging.warning(f"[{index:012d}] center bbox {cw}x{ch} != orig {w}x{h}; resizing paste")
                        out_img.paste(img.resize((cw, ch), Image.BICUBIC), (left, top))
                    else:
                        out_img.paste(img, (left, top))

                out_name = out_subdir / f"{index:012d}_cap{j:02d}_g{g:04.2f}.jpg"
                try:
                    out_img.save(out_name, quality=95)
                except Exception as e:
                    logging.warning(f"[{index:012d}] save fail cap#{j} g={g}: {e}")

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch SDXL Outpainting with per-image captions (multi-GPU).")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "tune"])
    parser.add_argument("--tune_indices", type=str, default="")
    parser.add_argument("--tune_file", type=Path, default=None)

    parser.add_argument("--val_img_folder", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)

    parser.add_argument("--model_id", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--vae_id", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpus", type=str, default="", help='Comma-separated GPU ids, e.g. "0,1,2,3". Empty=auto-all.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=str, default="7.5", help="CSV accepted, e.g. '2.5,5.0,7.5'")
    parser.add_argument("--guidence", type=str, default=None, help="Alias for --guidance")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mask_blur", type=int, default=24)
    parser.add_argument("--max_captions", type=int, default=10)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--paste_original_center", action="store_true")
    parser.add_argument("--save_inputs", action="store_true")
    # Working-size
    parser.add_argument("--work_long_side", type=int, default=1024)
    parser.add_argument("--work_multiple", type=int, default=8)
    # Verbosity
    parser.add_argument("-v", "--verbose", action="count", default=1)
    
    parser.add_argument(
        "--work_indices", type=str, default="",
        help='Further limit work to these indices/ranges, e.g. "0-24999,30010-35000".'
    )
    parser.add_argument(
        "--work_file", type=Path, default=None,
        help="Optional file with indices/ranges (one per line, # comments allowed) to further limit work."
    )
    parser.add_argument(
        "--shard", type=str, default="",
        help='Optional cross-machine sharding in the form "R/N" (e.g., 0/4). Uses (index %% N)==R.'
    )
    args = parser.parse_args()

    # Main-process logger
    setup_logger(args.verbose, prefix="MAIN")

    # Parse guidance list (CSV)
    guidance_spec = args.guidence if args.guidence is not None else args.guidance
    try:
        guidance_values = parse_float_list(guidance_spec)
    except Exception as e:
        logging.error(f"Failed to parse --guidance/--guidence: {e}"); return
    if not guidance_values:
        guidance_values = [7.5]
    logging.info(f"Guidance scales: {', '.join(f'{g:.2f}' for g in guidance_values)}")

    # Load captions (once to validate)
    id_to_caps = load_captions_jsonl(args.captions)
    if not id_to_caps:
        logging.error("No valid captions found in JSONL."); return

    # Collect images in folder
    img_paths_all, all_candidates, rejected = collect_indexed_images(args.val_img_folder, required_len=12)
    if not img_paths_all:
        log_no_match_diagnostics(args.val_img_folder, all_candidates, rejected, required_len=12); return
    index_to_path: Dict[int, Path] = {int(p.stem): p for p in img_paths_all}

    # Determine selected indices
    if args.mode == "all":
        selected_indices = sorted(index_to_path.keys())
    else:
        chosen: Set[int] = set()
        if args.tune_indices:
            try: chosen |= parse_indices_spec(args.tune_indices)
            except ValueError as e:
                logging.error(f"--tune_indices parse error: {e}"); return
        if args.tune_file is not None:
            if not args.tune_file.exists():
                logging.error(f"--tune_file not found: {args.tune_file}"); return
            chosen |= parse_indices_file(args.tune_file)
        if not chosen:
            logging.error("Tune mode selected but no indices provided."); return
        missing = sorted([i for i in chosen if i not in index_to_path])
        if missing:
            logging.warning(f"{len(missing)} tune indices have no matching %012d.jpg (e.g., {missing[:10]}).")
        selected_indices = sorted([i for i in chosen if i in index_to_path])
        if not selected_indices:
            logging.error("After filtering, no tune indices match any files."); return
        logging.info(f"Tune mode: {len(selected_indices)} indices to process.")
        
    # ---- Cross-machine filtering: work_indices / work_file ----
    work_set = set()
    if args.work_indices:
        try:
            work_set |= parse_indices_spec(args.work_indices)
        except ValueError as e:
            logging.error(f"--work_indices parse error: {e}")
            return
    if args.work_file is not None:
        if not args.work_file.exists():
            logging.error(f"--work_file not found: {args.work_file}")
            return
        work_set |= parse_indices_file(args.work_file)

    if work_set:
        before = len(selected_indices)
        selected_indices = sorted([i for i in selected_indices if i in work_set])
        logging.info(f"Applied work_indices/work_file: {before} -> {len(selected_indices)} indices")

    # ---- Optional modulo sharding across machines: --shard R/N ----
    try:
        shard = parse_shard(args.shard)
    except ValueError as e:
        logging.error(str(e))
        return

    # Use modulo on the actual numeric index (stable even if files are missing)
    if shard is not None:
        rank, world = shard
        before = len(selected_indices)
        selected_indices = sorted([i for i in selected_indices if (i % world) == rank])
        logging.info(f"Applied shard {rank}/{world}: {before} -> {len(selected_indices)} indices")

    if not selected_indices:
        logging.error("No indices left to process after work filtering/sharding.")
        return
    
    logging.info(f"indices sample: {selected_indices[:5]} ... {selected_indices[-5:]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Multi-GPU planning
    want_cuda = (args.device == "cuda")
    try:
        gpu_ids = parse_gpus(args.gpus, want_cuda)
    except Exception as e:
        logging.error(f"GPU parsing error: {e}"); return

    # If CPU or single GPU, run inline (no processes) for simplicity
    if not gpu_ids or len(gpu_ids) == 1:
        device_str = "cpu" if not want_cuda or not torch.cuda.is_available() else f"cuda:{gpu_ids[0]}" if gpu_ids else "cuda"
        logging.info(f"Running single worker on {device_str}")
        wargs = WorkerArgs(
            gpu_id=(gpu_ids[0] if gpu_ids else -1),
            device_str=device_str,
            indices=selected_indices,
            out_dir=str(args.out_dir),
            val_img_folder=str(args.val_img_folder),
            captions_path=str(args.captions),
            model_id=args.model_id,
            vae_id=(args.vae_id if args.vae_id.strip() else None),
            steps=args.steps,
            strength=args.strength,
            mask_blur=args.mask_blur,
            negative_prompt=args.negative_prompt,
            paste_original_center=args.paste_original_center,
            save_inputs=args.save_inputs,
            base_seed=(None if args.seed is None or args.seed < 0 else args.seed),
            verbose=args.verbose,
            work_long_side=args.work_long_side,
            work_multiple=args.work_multiple,
            guidance_values=guidance_values,
        )
        worker_main(wargs)
        logging.info("Done.")
        return

    # Multi-GPU: stride-assign indices
    n = len(gpu_ids)
    per_gpu_indices = [selected_indices[i::n] for i in range(n)]
    for i, lst in enumerate(per_gpu_indices):
        logging.info(f"GPU{gpu_ids[i]} gets {len(lst)} indices")

    # Spawn workers (one per GPU)
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    _install_signal_handlers()
    procs: List[Process] = []
    for i, gid in enumerate(gpu_ids):
        wargs = WorkerArgs(
            gpu_id=gid,
            device_str=f"cuda:{gid}",
            indices=per_gpu_indices[i],
            out_dir=str(args.out_dir),
            val_img_folder=str(args.val_img_folder),
            captions_path=str(args.captions),
            model_id=args.model_id,
            vae_id=(args.vae_id if args.vae_id.strip() else None),
            steps=args.steps,
            strength=args.strength,
            mask_blur=args.mask_blur,
            negative_prompt=args.negative_prompt,
            paste_original_center=args.paste_original_center,
            save_inputs=args.save_inputs,
            base_seed=(None if args.seed is None or args.seed < 0 else args.seed),
            verbose=args.verbose,
            work_long_side=args.work_long_side,
            work_multiple=args.work_multiple,
            guidance_values=guidance_values,
        )
        p = Process(target=worker_main, args=(wargs,), daemon=False)
        p.start(); procs.append(p)
        _CHILD_PROCS.append(p)

    # Join
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        _terminate_children("KeyboardInterrupt")
        raise

    logging.info("All workers finished.")

if __name__ == "__main__":
    main()
