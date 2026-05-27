import re
import os
import json
import io
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _resolve_model_class(pretrained: str, is_moe: bool):
    """Auto-detect and return the appropriate HF model class for a Qwen3 variant.

    Returns (model_class, dtype_kwarg_name) where dtype_kwarg_name is the
    keyword argument name for specifying dtype in from_pretrained().
    """
    config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if "qwen3_5" in model_type:
        from transformers import (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        )

        model_cls = Qwen3_5MoeForConditionalGeneration if is_moe else Qwen3_5ForConditionalGeneration
        dtype_key = "torch_dtype"
    else:
        if is_moe:
            try:
                from lmms_eval.experiment_models.qwen3_vl_experiments import Qwen3VLMoeForConditionalGeneration

                model_cls = Qwen3VLMoeForConditionalGeneration
            except ImportError as exc:
                raise ImportError(
                    "MoE Qwen3-VL checkpoints are not supported by the local 'qwen3_vl_experiments' fork "
                    "(missing Qwen3VLMoeForConditionalGeneration). Use a dense Qwen3-VL checkpoint or extend the "
                    "experiments fork with a MoE implementation."
                ) from exc
        else:
            from lmms_eval.experiment_models.qwen3_vl_experiments import Qwen3VLForConditionalGeneration

            model_cls = Qwen3VLForConditionalGeneration
        dtype_key = "dtype"

    return model_cls, dtype_key


def _best_factor_pair(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    r = int(n**0.5)
    for a in range(r, 0, -1):
        if n % a == 0:
            return a, n // a
    return 1, n


def _to_uint8_video(frames) -> Optional[np.ndarray]:
    if frames is None:
        return None
    if isinstance(frames, torch.Tensor):
        arr = frames.detach().cpu().numpy()
    else:
        arr = np.asarray(frames)

    # Expect (T, H, W, C) or (T, C, H, W)
    if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        maxv = float(np.nanmax(arr_f)) if arr_f.size else 0.0
        minv = float(np.nanmin(arr_f)) if arr_f.size else 0.0
        if maxv <= 1.0 and minv >= 0.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
    return arr


def _images_to_uint8_video(images) -> Optional[np.ndarray]:
    if images is None:
        return None
    # images may be a list (possibly nested) of PIL Images/arrays or a tensor
    if torch.is_tensor(images):
        arr = images.detach().cpu().numpy()
        if arr.ndim == 3:
            arr = arr[None, ...]
        return _to_uint8_video(arr)

    if isinstance(images, (list, tuple)):
        # Flatten nested lists/tuples
        def _flatten(seq):
            out = []
            for item in seq:
                if isinstance(item, (list, tuple)):
                    out.extend(_flatten(item))
                else:
                    out.append(item)
            return out

        flat = _flatten(images)
        frames = []
        for im in flat:
            if im is None:
                continue
            if torch.is_tensor(im):
                arr = im.detach().cpu().numpy()
            else:
                arr = np.asarray(im)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            frames.append(arr)
        if not frames:
            return None
        try:
            stacked = np.stack(frames, axis=0)
        except Exception:
            return None
        return _to_uint8_video(stacked)

    return None


def _doc_images_to_uint8_video(doc: Optional[dict]) -> Optional[np.ndarray]:
    if not isinstance(doc, dict):
        return None
    images = doc.get("images")
    single = doc.get("image")
    if isinstance(images, (list, tuple)):
        iter_images = list(images)
    elif single is not None:
        iter_images = [single]
    else:
        return None

    frames = []
    for im in iter_images:
        if im is None:
            continue
        if isinstance(im, Image.Image):
            arr = np.asarray(im.convert("RGB"))
        else:
            try:
                arr = np.asarray(Image.open(io.BytesIO(im)).convert("RGB"))
            except Exception:
                continue
        frames.append(arr)
    if not frames:
        return None
    try:
        stacked = np.stack(frames, axis=0)
    except Exception:
        return None
    return _to_uint8_video(stacked)


def _ensure_uint8_rgb(frames_uint8: np.ndarray) -> np.ndarray:
    arr = frames_uint8
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D video array (T,H,W,C); got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected C=3; got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _try_write_mp4(path: Path, frames_uint8: np.ndarray, fps: Optional[float] = None) -> tuple[bool, Optional[str]]:
    """Best-effort mp4 writer.

    Prefers OpenCV (available in the cluster env), falls back to imageio if present.
    Returns (ok, error_message).
    """

    fps_val = float(fps) if fps is not None else 30.0

    ffmpeg_err = None
    # Prefer ffmpeg H.264 (yuv420p) for broad playback compatibility (incl. VS Code preview).
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        try:
            rgb = _ensure_uint8_rgb(frames_uint8)
            h, w = int(rgb.shape[1]), int(rgb.shape[2])
            path.parent.mkdir(parents=True, exist_ok=True)

            cmd = [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps_val),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                str(path),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert proc.stdin is not None
            for frame in rgb:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            _, err = proc.communicate()
            if proc.returncode != 0:
                ffmpeg_err = err.decode("utf-8", errors="ignore")[-2000:]
            else:
                return True, None
        except Exception as exc:
            ffmpeg_err = str(exc)
    try:
        import cv2  # type: ignore

        rgb = _ensure_uint8_rgb(frames_uint8)
        h, w = int(rgb.shape[1]), int(rgb.shape[2])
        path.parent.mkdir(parents=True, exist_ok=True)

        # OpenCV expects BGR
        fourcc_candidates = ["mp4v", "avc1", "H264"]
        writer = None
        last_err = None
        for fourcc_str in fourcc_candidates:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(str(path), fourcc, fps_val, (w, h), True)
                if writer is not None and writer.isOpened():
                    last_err = None
                    break
                if writer is not None:
                    writer.release()
                writer = None
            except Exception as exc:
                last_err = str(exc)
                writer = None

        if writer is None:
            return False, last_err or "cv2.VideoWriter could not be opened"

        for frame in rgb:
            bgr = frame[..., ::-1]
            writer.write(bgr)
        writer.release()
        return True, None
    except Exception as exc:
        # Fallback: imageio (if installed)
        try:
            import imageio.v2 as imageio  # type: ignore

            path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(str(path), fps=fps_val) as writer:
                for frame in _ensure_uint8_rgb(frames_uint8):
                    writer.append_data(frame)
            return True, None
        except Exception as exc2:
            ffmpeg_part = f"ffmpeg error: {ffmpeg_err}; " if ffmpeg_err else ""
            msg = f"{ffmpeg_part}cv2 error: {exc}; imageio error: {exc2}"
            eval_logger.warning(f"Failed to write mp4 '{path}': {msg}")
            return False, msg


def _parse_fps_value(raw: str) -> Optional[float]:
    val = (raw or "").strip()
    if not val or val == "0/0":
        return None
    if "/" in val:
        num, denom = val.split("/", 1)
        try:
            n = float(num)
            d = float(denom)
        except ValueError:
            return None
        if d > 0:
            return n / d
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _estimate_video_fps(video_path: Optional[str]) -> Optional[float]:
    if not video_path:
        return None
    path = Path(video_path)
    if not path.exists():
        return None

    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        for key in ("avg_frame_rate", "r_frame_rate"):
            cmd = [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                f"stream={key}",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(path),
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                fps = _parse_fps_value(proc.stdout.splitlines()[0] if proc.stdout else "")
                if fps:
                    return fps
            except Exception:
                pass

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if fps and fps > 0:
            return fps
    except Exception:
        return None

    return None


def _normalize_to_uint8_heatmap(attn: np.ndarray) -> np.ndarray:
    # attn: (T, H, W) float in [0, 1] or unnormalized
    a = attn.astype(np.float32)
    if np.nanmax(a) > 1.0 or np.nanmin(a) < 0.0:
        a = a - np.nanmin(a)
        denom = np.nanmax(a)
        if denom > 0:
            a = a / denom
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    # grayscale -> RGB
    return np.repeat(a[..., None], 3, axis=-1)


def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Return start index of needle inside haystack, or None."""
    if not needle or not haystack or len(needle) > len(haystack):
        return None
    # Simple O(n*m) search; sequences are short enough for our use.
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def _split_words(text: str) -> List[str]:
    # Keep punctuation attached to words (matches how users read the question)
    return re.findall(r"\S+", text or "")


def _extract_options_from_prompt(prompt_text: Optional[str]) -> List[str]:
    """Extract MCA option lines (e.g., 'A. ...') from a rendered prompt."""
    if not isinstance(prompt_text, str) or not prompt_text:
        return []
    options: List[str] = []
    for line in prompt_text.splitlines():
        s = line.strip()
        if re.match(r"^[A-D][\.)]\s+\S+", s):
            options.append(s)
    # Keep only first 4 to avoid odd prompts.
    return options[:4]


def _aggregate_token_attn_to_words(
    tokenizer,
    question_text: str,
    question_token_ids: List[int],
    token_attn: List[float],
) -> Tuple[List[str], List[float]]:
    """Approximate mapping from token-level attention to word-level attention.

    Strategy: split question by whitespace into words; tokenize each word and
    greedily match its token IDs inside question_token_ids.
    """

    words = _split_words(question_text)
    if not words or not question_token_ids or not token_attn or len(question_token_ids) != len(token_attn):
        return words, [0.0 for _ in words]

    w_attn: List[float] = []
    pos = 0
    for w in words:
        w_ids = tokenizer.encode(w, add_special_tokens=False)
        if not w_ids:
            w_attn.append(0.0)
            continue
        # Try to match starting at current position, else search forward.
        start = _find_subsequence(question_token_ids[pos:], w_ids)
        if start is None:
            # fallback: try with stripped punctuation variants
            w_stripped = re.sub(r"(^[^\w]+|[^\w]+$)", "", w)
            if w_stripped and w_stripped != w:
                w_ids2 = tokenizer.encode(w_stripped, add_special_tokens=False)
                start = _find_subsequence(question_token_ids[pos:], w_ids2) if w_ids2 else None
                if start is not None:
                    w_ids = w_ids2
        if start is None:
            w_attn.append(0.0)
            continue
        start = pos + start
        end = start + len(w_ids)
        end = min(end, len(token_attn))
        w_attn.append(float(sum(token_attn[start:end])))
        pos = end

    # Normalize to [0,1] for visualization
    mx = max(w_attn) if w_attn else 0.0
    if mx > 0:
        w_attn = [float(x) / float(mx) for x in w_attn]
    return words, w_attn


def _save_question_word_attn_png(path: Path, question_text: str, words: List[str], weights_01: List[float]) -> tuple[bool, Optional[str]]:
    """Save a word-level attention heatmap.

    Each word is drawn on a colored rectangle; the rectangle color encodes attention weight.
    Weights are (re)normalized to [0,1] inside this function.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Normalize weights defensively
        ws = []
        for a in (weights_01 or []):
            a = float(a)
            if not (a == a):
                a = 0.0
            ws.append(a)
        if ws:
            mn = min(ws)
            mx = max(ws)
            if mx > mn:
                ws = [(x - mn) / (mx - mn) for x in ws]
            else:
                ws = [0.0 for _ in ws]
        else:
            ws = []

        words = words or _split_words(question_text)
        if len(ws) != len(words):
            # If mismatch, pad/truncate with zeros.
            ws = (ws + [0.0] * len(words))[: len(words)]

        # Layout
        outer_pad = 20
        cell_pad_x = 10
        cell_pad_y = 6
        cell_gap = 6
        max_w = 1400
        bg = (255, 255, 255)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # First pass: compute wrapped layout and final height
        dummy = Image.new("RGB", (max_w, 10), color=bg)
        ddraw = ImageDraw.Draw(dummy)

        x = outer_pad
        y = outer_pad
        line_h = 0
        placements: list[tuple[str, float, int, int, int, int]] = []  # word, weight, x0, y0, x1, y1

        for w, a in zip(words, ws):
            w_txt = str(w)
            bbox = ddraw.textbbox((0, 0), w_txt, font=font)
            text_w = int(bbox[2] - bbox[0])
            text_h = int(bbox[3] - bbox[1])
            cell_w = text_w + 2 * cell_pad_x
            cell_h = text_h + 2 * cell_pad_y

            if x + cell_w > max_w - outer_pad:
                x = outer_pad
                y += max(line_h, cell_h) + cell_gap
                line_h = 0

            x0, y0 = x, y
            x1, y1 = x + cell_w, y + cell_h
            placements.append((w_txt, float(a), x0, y0, x1, y1))
            x = x1 + cell_gap
            line_h = max(line_h, cell_h)

        used_w = 0
        if placements:
            used_w = max(p[4] for p in placements) + outer_pad  # x1 + right pad
        final_w = int(max(240, min(max_w, used_w if used_w > 0 else max_w)))

        final_h = y + (line_h if placements else 0) + outer_pad
        final_h = int(max(final_h, 80))

        img = Image.new("RGB", (final_w, final_h), color=bg)
        draw = ImageDraw.Draw(img)

        def _color_from_weight(t: float) -> tuple[int, int, int]:
            # White (low) -> Red (high)
            t = 0.0 if not (t == t) else t
            t = max(0.0, min(1.0, float(t)))
            g = int(255 * (1.0 - t))
            b = int(255 * (1.0 - t))
            return (255, g, b)

        def _text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
            r, g, b = bg_rgb
            # Perceived luminance; choose black text unless bg is very dark
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return (0, 0, 0) if lum > 110 else (255, 255, 255)

        for w_txt, a, x0, y0, x1, y1 in placements:
            c = _color_from_weight(a)
            draw.rectangle([x0, y0, x1, y1], fill=c, outline=(220, 220, 220), width=1)
            bbox = draw.textbbox((0, 0), w_txt, font=font)
            text_w = int(bbox[2] - bbox[0])
            text_h = int(bbox[3] - bbox[1])
            tx = x0 + (x1 - x0 - text_w) // 2
            ty = y0 + (y1 - y0 - text_h) // 2
            draw.text((tx, ty), w_txt, fill=_text_color(c), font=font)

        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _aggregate_question_attn_to_words_from_offsets(
    *,
    tokenizer,
    prompt_text: str,
    question_text: str,
    question_char_span: tuple[int, int],
    question_token_indices_prompt: List[int],
    token_attn: List[float],
) -> tuple[List[str], List[float]]:
    """Aggregate token attention to words using character offsets.

    This avoids brittle token-id subsequence matching (which breaks with leading-space/BPE tokens).
    """

    words = _split_words(question_text)
    if not words or not question_text:
        return words, [0.0 for _ in words]
    if not question_token_indices_prompt or not token_attn or len(question_token_indices_prompt) != len(token_attn):
        return words, [0.0 for _ in words]

    q0, q1 = question_char_span
    q0 = int(max(0, q0))
    q1 = int(max(q0, q1))

    try:
        enc = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc.get("offset_mapping")
    except Exception:
        offsets = None
    if offsets is None:
        return words, [0.0 for _ in words]

    # Compute each word's char span within the full prompt
    word_spans: List[tuple[int, int]] = []
    pos = 0
    for w in words:
        idx = question_text.find(w, pos)
        if idx < 0:
            idx = question_text.find(w)
        if idx < 0:
            word_spans.append((q0, q0))
            continue
        ws = q0 + idx
        we = q0 + idx + len(w)
        word_spans.append((ws, we))
        pos = idx + len(w)

    w_attn = [0.0 for _ in words]
    for idx_prompt, a in zip(question_token_indices_prompt, token_attn):
        try:
            s, e = offsets[int(idx_prompt)]
        except Exception:
            continue
        if s is None or e is None or e <= s:
            continue
        if e <= q0 or s >= q1:
            continue
        for wi, (ws, we) in enumerate(word_spans):
            if ws < e and we > s:
                w_attn[wi] += float(a)

    mx = max(w_attn) if w_attn else 0.0
    if mx > 0:
        w_attn = [float(x) / float(mx) for x in w_attn]
    return words, w_attn


def _save_options_word_attn_png(
    path: Path,
    option_texts: List[str],
    option_words: List[List[str]],
    option_weights: List[List[float]],
) -> tuple[bool, Optional[str]]:
    """Save an options heatmap PNG by stacking per-option word heatmaps."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        if not option_texts or not option_words or not option_weights:
            return False, "no options"
        if not (len(option_texts) == len(option_words) == len(option_weights)):
            return False, "options length mismatch"

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Render each option line using the same word-heatmap logic as question
        rendered: List[Image.Image] = []
        gap_y = 12
        pad = 16

        for opt_text, words, weights in zip(option_texts, option_words, option_weights):
            # Reuse the question renderer by drawing into a temp file-like canvas
            # (call the same function but capture its output by re-implementing small wrapper)
            tmp_path = path.parent / (path.name + ".__tmp__.png")
            ok, err = _save_question_word_attn_png(tmp_path, opt_text, words, weights)
            if not ok:
                return False, err
            with Image.open(tmp_path) as im:
                img = im.convert("RGB")
            try:
                tmp_path.unlink(missing_ok=True)  # type: ignore[attr-defined]
            except Exception:
                pass
            # Add a small label bar above each option
            label_h = 22
            labeled = Image.new("RGB", (max(img.width, 240), img.height + label_h), color=(255, 255, 255))
            d = ImageDraw.Draw(labeled)
            d.text((pad, 2), str(opt_text)[:90], fill=(30, 30, 30), font=font)
            labeled.paste(img, (0, label_h))
            rendered.append(labeled)

        out_w = max(r.width for r in rendered) + 2 * pad
        out_h = sum(r.height for r in rendered) + gap_y * (len(rendered) - 1) + 2 * pad
        canvas = Image.new("RGB", (out_w, out_h), color=(255, 255, 255))
        y = pad
        for r in rendered:
            canvas.paste(r, (pad, y))
            y += r.height + gap_y

        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _normalize_attention_01(attn_map_thw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize attention map to [0,1] (per-clip) and return stats."""
    a = attn_map_thw.astype(np.float32)
    amin = float(np.nanmin(a)) if a.size else 0.0
    amax = float(np.nanmax(a)) if a.size else 0.0
    a = a - amin
    denom = float(np.nanmax(a)) if a.size else 0.0
    if denom > 0:
        a = a / denom
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    return a.clip(0.0, 1.0), {"attn_min": amin, "attn_max": amax}


def _resize_attn_to_frame(attn_hw01: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a single (h,w) attention map to (H,W) using bilinear."""
    import cv2  # type: ignore

    return cv2.resize(attn_hw01.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _overlay_attention(video_uint8_thwc: np.ndarray, attn_01_thw: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """Overlay a heatmap where attention is high.

    Output = (1-a)*orig + a*heat, where a = strength * attn.
    """
    rgb = _ensure_uint8_rgb(video_uint8_thwc)
    t, h, w, _ = rgb.shape
    out = np.empty_like(rgb)

    def _jet_colormap(x: np.ndarray) -> np.ndarray:
        # Approximate MATLAB jet: blue -> cyan -> yellow -> red
        x = np.clip(x, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
        return np.stack([r, g, b], axis=-1) * 255.0

    for i in range(t):
        a = attn_01_thw[min(i, attn_01_thw.shape[0] - 1)]
        a_big = _resize_attn_to_frame(a, h, w)
        alpha = (strength * a_big).clip(0.0, 1.0)[..., None]
        heat = _jet_colormap(a_big)
        out[i] = ((1.0 - alpha) * rgb[i].astype(np.float32) + alpha * heat).clip(0, 255).astype(np.uint8)
    return out

def _overlay_attention_red(video_uint8_thwc: np.ndarray, attn_01_thw: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """Overlay red where attention is high.

    Output = (1-a)*orig + a*red, where a = strength * attn.
    """
    rgb = _ensure_uint8_rgb(video_uint8_thwc)
    t, h, w, _ = rgb.shape
    out = np.empty_like(rgb)
    red = np.zeros((h, w, 3), dtype=np.float32)
    red[..., 0] = 255.0
    for i in range(t):
        a = attn_01_thw[min(i, attn_01_thw.shape[0] - 1)]
        a_big = _resize_attn_to_frame(a, h, w)
        alpha = (strength * a_big).clip(0.0, 1.0)[..., None]
        out[i] = ((1.0 - alpha) * rgb[i].astype(np.float32) + alpha * red).clip(0, 255).astype(np.uint8)
    return out


def _save_vertical_stack_png(path: Path, images_uint8: list[np.ndarray]) -> tuple[bool, Optional[str]]:
    """Stack multiple rollout images vertically into one PNG."""
    try:
        def _ensure_uint8_rgb_image(img: np.ndarray) -> np.ndarray:
            arr = np.asarray(img)
            if arr.ndim != 3:
                raise ValueError(f"Expected 3D image (H,W,C); got shape {arr.shape}")
            if arr.dtype != np.uint8:
                arr = arr.clip(0, 255).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.shape[-1] != 3:
                raise ValueError(f"Expected C=3; got shape {arr.shape}")
            return arr

        pil_images = [Image.fromarray(_ensure_uint8_rgb_image(img)) for img in images_uint8]
        widths = [im.size[0] for im in pil_images]
        heights = [im.size[1] for im in pil_images]
        W = max(widths)
        H = sum(heights)
        canvas = Image.new("RGB", (W, H))
        y = 0
        for im in pil_images:
            canvas.paste(im, (0, y))
            y += im.size[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _compute_vsibench_accuracy(doc: dict, prediction: str) -> tuple[Optional[float], dict]:
    """Compute per-sample score matching vsibench_process_results (approx).

    Returns (score, debug) where score is:
    - MCA: exact match accuracy in {0,1}
    - NA:  mean relative accuracy in [0,1]
    """
    debug: dict = {}
    try:
        from lmms_eval.tasks.vsibench.utils import (
            MCA_QUESTION_TYPES,
            NA_QUESTION_TYPES,
            fuzzy_matching,
            exact_match,
            mean_relative_accuracy,
            to_float,
        )

        gt = doc.get("ground_truth")
        qtype = doc.get("question_type")
        debug["ground_truth"] = gt
        debug["question_type"] = qtype
        pred = fuzzy_matching(prediction or "")
        if qtype in MCA_QUESTION_TYPES:
            return float(exact_match(pred, str(gt))), debug
        if qtype in NA_QUESTION_TYPES:
            try:
                return float(mean_relative_accuracy(to_float(pred), to_float(gt), start=0.5, end=0.95, interval=0.05)), debug
            except Exception:
                return 0.0, debug
        return None, debug
    except Exception as exc:
        debug["error"] = str(exc)
        return None, debug


_SAFE_PATH_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _safe_path_component(text: Optional[str], default: str = "unknown") -> str:
    s = (text or "").strip()
    if not s:
        return default
    s = _SAFE_PATH_RE.sub("_", s)
    s = s.strip("._-")
    return s or default


def _save_rollout_png(path: Path, frames_uint8_thwc: np.ndarray) -> tuple[bool, Optional[str]]:
    """Save frames concatenated horizontally as a single PNG."""
    try:
        from PIL import Image

        rgb = _ensure_uint8_rgb(frames_uint8_thwc)
        t, h, w, _ = rgb.shape
        canvas = Image.new("RGB", (w * t, h))
        for i in range(t):
            canvas.paste(Image.fromarray(rgb[i]), (i * w, 0))
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _save_experiment_artifacts(
    out_dir: Path,
    sample_tag: str,
    attn_map_thw: Optional[np.ndarray],
    video_uint8: Optional[np.ndarray],
    metadata: dict,
    save_mp4: bool = False,
    save_npz: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / sample_tag

    arrays: dict[str, np.ndarray] = {}
    attn_norm_01 = None
    attn_stats = None
    if attn_map_thw is not None:
        attn_norm_01, attn_stats = _normalize_attention_01(attn_map_thw)
        arrays["attn_map_thw"] = attn_map_thw.astype(np.float16)
        arrays["attn_norm_01_thw"] = attn_norm_01.astype(np.float16)
    if video_uint8 is not None:
        arrays["video_uint8_thwc"] = video_uint8
    if save_npz and arrays:
        np.savez_compressed(base.with_suffix(".npz"), **arrays)

    video_ok = None
    video_err = None
    if save_mp4 and video_uint8 is not None:
        video_ok, video_err = _try_write_mp4(base.with_suffix(".mp4"), video_uint8, fps=metadata.get("video_fps"))

    rollout_ok = None
    rollout_err = None
    if video_uint8 is not None:
        rollout_ok, rollout_err = _save_rollout_png(base.with_name(base.name + "_rollout.png"), video_uint8)

    attn_ok = None
    attn_err = None
    attn_rollout_ok = None
    attn_rollout_err = None
    if attn_norm_01 is not None:
        heat = _normalize_to_uint8_heatmap(attn_norm_01)
        if save_mp4:
            attn_ok, attn_err = _try_write_mp4(base.with_name(base.name + "_attn.mp4"), heat, fps=metadata.get("video_fps"))
        attn_rollout_ok, attn_rollout_err = _save_rollout_png(base.with_name(base.name + "_attn_rollout.png"), heat)

    masked_ok = None
    masked_err = None
    masked_rollout_ok = None
    masked_rollout_err = None
    mask_rollout_ok = None
    mask_rollout_err = None
    stacked_rollout_ok = None
    stacked_rollout_err = None
    if video_uint8 is not None and attn_norm_01 is not None:
        overlay = _overlay_attention(video_uint8, attn_norm_01)
        if save_mp4:
            masked_ok, masked_err = _try_write_mp4(base.with_name(base.name + "_masked.mp4"), overlay, fps=metadata.get("video_fps"))
        masked_rollout_ok, masked_rollout_err = _save_rollout_png(base.with_name(base.name + "_masked_rollout.png"), overlay)

        # Also save the attention mask itself as rollout (heatmap already computed above)
        heat = _normalize_to_uint8_heatmap(attn_norm_01)
        mask_rollout_ok, mask_rollout_err = _save_rollout_png(base.with_name(base.name + "_mask_rollout.png"), heat)

        # Vertical stack of rollouts: (original, heatmask, masked)
        def _rollout_array(frames_thwc: np.ndarray) -> np.ndarray:
            frames = _ensure_uint8_rgb(frames_thwc)
            t, h, w, _ = frames.shape
            canvas = Image.new("RGB", (w * t, h))
            for i in range(t):
                canvas.paste(Image.fromarray(frames[i]), (i * w, 0))
            return np.asarray(canvas)

        orig_roll = _rollout_array(video_uint8)
        mask_roll = _rollout_array(heat)
        masked_roll = _rollout_array(overlay)
        stacked_rollout_ok, stacked_rollout_err = _save_vertical_stack_png(
            base.with_name(base.name + "_stacked_rollout.png"),
            [orig_roll, mask_roll, masked_roll],
        )

    metadata = dict(metadata)
    if attn_stats is not None:
        metadata.update(attn_stats)
    metadata["write_video_mp4_ok"] = video_ok
    metadata["write_video_mp4_error"] = video_err
    metadata["save_mp4"] = bool(save_mp4)
    metadata["save_npz"] = bool(save_npz)
    metadata["write_video_rollout_png_ok"] = rollout_ok
    metadata["write_video_rollout_png_error"] = rollout_err
    metadata["write_attn_mp4_ok"] = attn_ok
    metadata["write_attn_mp4_error"] = attn_err
    metadata["write_attn_rollout_png_ok"] = attn_rollout_ok
    metadata["write_attn_rollout_png_error"] = attn_rollout_err
    metadata["write_masked_mp4_ok"] = masked_ok
    metadata["write_masked_mp4_error"] = masked_err
    metadata["write_masked_rollout_png_ok"] = masked_rollout_ok
    metadata["write_masked_rollout_png_error"] = masked_rollout_err
    metadata["write_mask_rollout_png_ok"] = mask_rollout_ok
    metadata["write_mask_rollout_png_error"] = mask_rollout_err
    metadata["write_stacked_rollout_png_ok"] = stacked_rollout_ok
    metadata["write_stacked_rollout_png_error"] = stacked_rollout_err

    (base.with_suffix(".json")).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    # Question word-attention visualization
    q_ok = None
    q_err = None
    q_text = metadata.get("question_text")
    q_words = metadata.get("question_words")
    q_attn = metadata.get("question_word_attn")
    if isinstance(q_text, str) and isinstance(q_words, list) and isinstance(q_attn, list) and len(q_words) == len(q_attn):
        q_ok, q_err = _save_question_word_attn_png(base.with_name(base.name + "_question_attn.png"), q_text, q_words, q_attn)
        # update JSON with write status
        try:
            meta2 = dict(metadata)
            meta2["write_question_attn_png_ok"] = q_ok
            meta2["write_question_attn_png_error"] = q_err
            (base.with_suffix(".json")).write_text(json.dumps(meta2, indent=2, ensure_ascii=False))
        except Exception:
            pass

    # Options word-attention visualization (MCA prompts)
    opt_ok = None
    opt_err = None
    opt_texts = metadata.get("options_texts")
    opt_words = metadata.get("option_words")
    opt_attn = metadata.get("option_word_attn")
    if (
        isinstance(opt_texts, list)
        and isinstance(opt_words, list)
        and isinstance(opt_attn, list)
        and len(opt_texts) == len(opt_words) == len(opt_attn)
    ):
        # Validate each option
        ok_shape = True
        for w, a in zip(opt_words, opt_attn):
            if not (isinstance(w, list) and isinstance(a, list) and len(w) == len(a)):
                ok_shape = False
                break
        if ok_shape and len(opt_texts) > 0:
            opt_ok, opt_err = _save_options_word_attn_png(
                base.with_name(base.name + "_options_attn.png"),
                [str(x) for x in opt_texts],
                opt_words,
                opt_attn,
            )
            try:
                meta3 = json.loads((base.with_suffix(".json")).read_text())
            except Exception:
                meta3 = dict(metadata)
            try:
                meta3["write_options_attn_png_ok"] = opt_ok
                meta3["write_options_attn_png_error"] = opt_err
                (base.with_suffix(".json")).write_text(json.dumps(meta3, indent=2, ensure_ascii=False))
            except Exception:
                pass


@register_model("qwen3_vl_experiments")
class Qwen3_VL_Experiments(lmms):
    """
    Unified model class for the Qwen3-VL and Qwen3.5 model families (Experiments variant).

    Auto-detects the model variant from the HuggingFace config and loads
    the appropriate model class. Supports both dense and MoE variants.

    - Qwen3-VL: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
    - Qwen3.5:  https://huggingface.co/Qwen/Qwen3.5-4B
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = None,
        reasoning_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # For attention logging we need real attention weights.
        # SDPA/flash kernels frequently return None for attention weights even when
        # output_attentions=True. We use a step-wise capture path (no prefill
        # attentions), so enabling eager here is typically safe memory-wise.
        save_attn_env = os.environ.get("LMMS_EVAL_EXPERIMENTS_SAVE_ATTN", "0")
        save_attn_enabled = save_attn_env not in ("", "0", "false", "False")
        if save_attn_enabled and attn_implementation is None:
            attn_implementation = "eager"

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Auto-detect model variant and load the appropriate HF class
        is_moe = bool(re.search(r"A\d+B", pretrained))
        model_cls, dtype_key = _resolve_model_class(pretrained, is_moe)

        model_kwargs = {
            dtype_key: "bfloat16",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = model_cls.from_pretrained(pretrained, **model_kwargs).eval()
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.enable_thinking = enable_thinking

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3 VL models")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _build_video_kwargs(self):
        """Build video processing kwargs based on model configuration."""
        video_kwargs = {"min_pixels": self.min_pixels}

        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        elif self.total_pixels is not None:
            video_kwargs["max_frames"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames

        if self.total_pixels is not None:
            video_kwargs["total_pixels"] = self.total_pixels
        else:
            video_kwargs["max_pixels"] = self.max_pixels

        return video_kwargs

    def _apply_chat_template(self, batched_messages, **kwargs):
        """Apply chat template with optional enable_thinking support."""
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        template_kwargs.update(kwargs)
        return self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True, **template_kwargs)

    def _build_generate_kwargs(self, gen_kwargs):
        """Build model.generate() kwargs from user gen_kwargs merged with defaults."""
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        pad_token_id = self.tokenizer.pad_token_id

        if current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)

        generate_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": current["do_sample"],
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            val = current.get(key)
            if val is not None:
                generate_kwargs[key] = val

        return generate_kwargs

    def _strip_thinking(self, answer):
        """Strip <think>...</think> content from model output if enable_thinking is set."""
        if self.enable_thinking:
            _, _, remaining = answer.partition("</think>")
            return remaining.strip()
        return answer

    def _preprocess_chunk(self, chunk, return_media: bool = False):
        """Preprocess a batch chunk on CPU: message building, video decoding, tokenization.

        Returns (inputs, contexts, gen_kwargs, until) with inputs still on CPU.
        """
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        visual_list = [doc_to_visual[0](self.task_dict[t][s][i]) for t, s, i in zip(task, split, doc_id)]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")
        until = [item for item in until if item != "\n\n"]

        if isinstance(contexts, tuple):
            contexts = list(contexts)

        for i in range(len(contexts)):
            if "<image>" in contexts[i]:
                contexts[i] = contexts[i].replace("<image>", "")

        video_kwargs = self._build_video_kwargs()

        batched_messages = []
        for i, context in enumerate(contexts):
            if "<image>" in context:
                context = context.replace("<image>", "")

            message = [{"role": "system", "content": self.system_prompt}]

            if self.reasoning_prompt:
                context = context.strip() + self.reasoning_prompt
                contexts[i] = context

            processed_visuals = []
            if visual_list[i] is not None:
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        processed_visuals.append(
                            {
                                "type": "video",
                                "video": visual,
                                **video_kwargs,
                            }
                        )
                    elif isinstance(visual, Image.Image):
                        processed_visuals.append(
                            {
                                "type": "image",
                                "image": visual,
                                "max_pixels": self.max_pixels,
                                "min_pixels": self.min_pixels,
                            }
                        )

            if self.interleave_visuals is False:
                message.append(
                    {
                        "role": "user",
                        "content": processed_visuals + [{"type": "text", "text": context}],
                    }
                )
            else:
                image_placeholders = re.findall(r"<image \d+>", context)
                content_parts = []
                text_parts = re.split(r"<image \d+>", context)
                if text_parts[0]:
                    content_parts.append({"type": "text", "text": text_parts[0]})

                for placeholder_idx, placeholder in enumerate(image_placeholders):
                    img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                    image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                    if processed_visuals and image_idx < len(processed_visuals):
                        content_parts.append(processed_visuals[image_idx])
                    if placeholder_idx + 1 < len(text_parts) and text_parts[placeholder_idx + 1]:
                        content_parts.append({"type": "text", "text": text_parts[placeholder_idx + 1]})

                message.append(
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                )

            batched_messages.append(message)

        texts = self._apply_chat_template(batched_messages)
        image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
            batched_messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        video_metadata_list = None
        if video_inputs is not None:
            video_inputs, video_metadata_list = map(list, zip(*video_inputs))

        if self.batch_size > 1:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                **processed_video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )

        if return_media:
            return inputs, contexts, gen_kwargs, until, texts, image_inputs, video_inputs, video_metadata_list
        return inputs, contexts, gen_kwargs, until

    def _locate_vision_token_indices(self, input_ids: torch.Tensor) -> tuple[list[torch.Tensor], dict]:
        """Locate vision-token key positions inside the prompt input_ids."""

        debug: dict = {
            "vision_locator": None,
            "vision_token_counts": None,
        }

        config = getattr(self.model, "config", None)
        image_token_id = getattr(config, "image_token_id", None)
        video_token_id = getattr(config, "video_token_id", None)
        vision_start_token_id = getattr(config, "vision_start_token_id", None)
        vision_end_token_id = getattr(config, "vision_end_token_id", None)

        idx_lists: list[torch.Tensor] = []
        if image_token_id is not None or video_token_id is not None:
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            if image_token_id is not None:
                mask |= input_ids.eq(int(image_token_id))
            if video_token_id is not None:
                mask |= input_ids.eq(int(video_token_id))
            if mask.any():
                debug["vision_locator"] = "image_token_id/video_token_id"
                for b in range(input_ids.shape[0]):
                    idx_lists.append(torch.nonzero(mask[b], as_tuple=False).squeeze(-1))

        if not idx_lists and vision_start_token_id is not None and vision_end_token_id is not None:
            debug["vision_locator"] = "vision_start->vision_end"
            vs = int(vision_start_token_id)
            ve = int(vision_end_token_id)
            for b in range(input_ids.shape[0]):
                ids = input_ids[b]
                starts = torch.nonzero(ids.eq(vs), as_tuple=False).squeeze(-1)
                ends = torch.nonzero(ids.eq(ve), as_tuple=False).squeeze(-1)
                seg_idxs = []
                if starts.numel() and ends.numel():
                    for s in starts.tolist():
                        e_candidates = ends[ends > s]
                        if e_candidates.numel() == 0:
                            continue
                        e = int(e_candidates[0].item())
                        if e > s + 1:
                            seg_idxs.append(torch.arange(s + 1, e, device=ids.device))
                if seg_idxs:
                    idx_lists.append(torch.cat(seg_idxs))
                else:
                    idx_lists.append(ids.new_zeros((0,), dtype=torch.long))

        if not idx_lists:
            debug["vision_locator"] = "none"
            idx_lists = [input_ids.new_zeros((0,), dtype=torch.long) for _ in range(input_ids.shape[0])]

        debug["vision_token_counts"] = [int(x.numel()) for x in idx_lists]
        return idx_lists, debug

    def _avg_attn_vec_to_vision_from_step(self, step_attentions, idx_lists: list[torch.Tensor]) -> tuple[Optional[torch.Tensor], dict]:
        """Compute (B, max_n_vision) attention-to-vision for a single decoded token step.

        step_attentions: expected tuple(layers) of (B, H, 1, S) or (B, H, S).
        """

        debug = {"attn_layers": None, "no_valid_attention_tensors": False, "none_layer_count": 0}
        if step_attentions is None:
            debug["no_valid_attention_tensors"] = True
            return None, debug

        layer_tensors = []
        for layer_attn in (step_attentions if isinstance(step_attentions, (tuple, list)) else [step_attentions]):
            if layer_attn is None:
                debug["none_layer_count"] += 1
                continue
            if not torch.is_tensor(layer_attn):
                continue
            if layer_attn.ndim == 3:
                layer_attn = layer_attn.unsqueeze(-2)
            if layer_attn.ndim != 4:
                continue
            layer_tensors.append(layer_attn)

        if not layer_tensors:
            debug["no_valid_attention_tensors"] = True
            return None, debug

        debug["attn_layers"] = len(layer_tensors)
        layers = torch.stack(layer_tensors, dim=0)  # (L, B, H, 1, S)
        q = layers[:, :, :, -1, :].mean(dim=0).mean(dim=1)  # (B, S)

        max_n = max(int(x.numel()) for x in idx_lists)
        out = q.new_zeros((q.shape[0], max_n))
        for b, idx in enumerate(idx_lists):
            if idx.numel() == 0:
                continue
            out[b, : idx.numel()] = q[b, idx]
        return out, debug

    def _avg_attn_vec_from_step(self, step_attentions, idx_lists: list[torch.Tensor]) -> tuple[Optional[torch.Tensor], dict]:
        """Generic gather: (B, max_n) attention-to-selected-tokens for a single step."""
        return self._avg_attn_vec_to_vision_from_step(step_attentions, idx_lists)

    def _locate_question_token_indices(self, input_ids: torch.Tensor, question_texts: List[str]) -> tuple[list[torch.Tensor], dict]:
        return self._locate_question_token_indices_from_prompt(input_ids, question_texts, prompt_texts=None)

    def _locate_question_token_indices_from_prompt(
        self,
        input_ids: torch.Tensor,
        question_texts: List[str],
        prompt_texts: Optional[List[str]],
    ) -> tuple[list[torch.Tensor], dict]:
        """Locate question token indices robustly.

        If prompt_texts is provided and tokenizer supports offset mapping, locate the question by
        character span inside the prompt and map to token indices.

        Fallback: token-subsequence search.
        """

        debug: dict = {"question_locator": None, "question_token_counts": None}
        idx_lists: list[torch.Tensor] = []

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        ids_cpu = input_ids.detach().cpu().tolist()

        for b, q in enumerate(question_texts):
            q = q or ""
            prompt = (prompt_texts[b] if prompt_texts is not None and b < len(prompt_texts) else "") or ""

            # Try offset-mapping path first
            if prompt and q:
                q_find = q
                start_char = prompt.find(q_find)
                if start_char < 0:
                    q_find = q.strip()
                    start_char = prompt.find(q_find)
                if start_char >= 0:
                    end_char = start_char + len(q_find)
                    try:
                        enc = self.tokenizer(
                            prompt,
                            add_special_tokens=False,
                            return_offsets_mapping=True,
                        )
                        offsets = enc.get("offset_mapping")
                        if offsets is not None:
                            # Determine left pad length in input_ids
                            seq = list(map(int, ids_cpu[b]))
                            pad_len = 0
                            if pad_id is not None:
                                for tid in seq:
                                    if tid == int(pad_id):
                                        pad_len += 1
                                    else:
                                        break
                            tok_idxs = []
                            for i, (s, e) in enumerate(offsets):
                                if s is None or e is None:
                                    continue
                                if e <= s:
                                    continue
                                if s < end_char and e > start_char:
                                    tok_idxs.append(pad_len + int(i))
                            if tok_idxs:
                                idx_lists.append(torch.tensor(tok_idxs, device=input_ids.device, dtype=torch.long))
                                continue
                    except Exception:
                        pass

            # Fallback: token-subsequence match
            seq = list(map(int, ids_cpu[b]))
            variants = [q, q.strip(), "\n" + q.strip(), " " + q.strip()]
            start = None
            needle = None
            for v in variants:
                cand = self.tokenizer.encode(v, add_special_tokens=False)
                if not cand:
                    continue
                s = _find_subsequence(seq, cand)
                if s is not None:
                    start = s
                    needle = cand
                    break
            if start is None or needle is None:
                idx_lists.append(input_ids.new_zeros((0,), dtype=torch.long))
                continue
            idx_lists.append(torch.arange(start, start + len(needle), device=input_ids.device))

        debug["question_locator"] = "offset_mapping" if prompt_texts is not None else "token_subsequence"
        debug["question_token_counts"] = [int(x.numel()) for x in idx_lists]
        return idx_lists, debug

    def _locate_option_token_indices_from_prompt(
        self,
        input_ids: torch.Tensor,
        prompt_texts: Optional[List[str]],
        option_texts_per_sample: Optional[List[List[str]]],
        max_options: int = 4,
    ) -> tuple[list[list[torch.Tensor]], dict]:
        """Return token indices for each option line.

        Output: idx_lists_per_option[opt_i][b] = 1D token indices in input_ids.
        """
        debug: dict = {"option_locator": None, "option_token_counts": None}
        bsz = int(input_ids.shape[0])
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        ids_cpu = input_ids.detach().cpu().tolist()

        empty = input_ids.new_zeros((0,), dtype=torch.long)
        idx_lists_per_option: list[list[torch.Tensor]] = [[empty for _ in range(bsz)] for _ in range(max_options)]

        if prompt_texts is None or option_texts_per_sample is None:
            debug["option_locator"] = "none"
            debug["option_token_counts"] = [[0 for _ in range(max_options)] for _ in range(bsz)]
            return idx_lists_per_option, debug

        counts: list[list[int]] = [[0 for _ in range(max_options)] for _ in range(bsz)]
        for b in range(bsz):
            prompt = prompt_texts[b] if b < len(prompt_texts) else ""
            opts = option_texts_per_sample[b] if b < len(option_texts_per_sample) else []
            if not isinstance(prompt, str) or not prompt:
                continue

            # Determine left pad length in input_ids
            seq = list(map(int, ids_cpu[b]))
            pad_len = 0
            if pad_id is not None:
                for tid in seq:
                    if tid == int(pad_id):
                        pad_len += 1
                    else:
                        break

            try:
                enc = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = enc.get("offset_mapping")
            except Exception:
                offsets = None

            if offsets is None:
                continue

            for oi in range(min(max_options, len(opts))):
                opt = opts[oi]
                if not isinstance(opt, str) or not opt:
                    continue
                start_char = prompt.find(opt)
                if start_char < 0:
                    opt2 = re.sub(r"\s+", " ", opt).strip()
                    prompt2 = re.sub(r"\s+", " ", prompt)
                    start_char = prompt2.find(opt2)
                    # Can't safely map back to original offsets if we changed text.
                    if start_char < 0:
                        continue
                    # Give up in this rare case.
                    continue
                end_char = start_char + len(opt)

                tok_idxs = []
                for i, (s, e) in enumerate(offsets):
                    if s is None or e is None or e <= s:
                        continue
                    if s < end_char and e > start_char:
                        tok_idxs.append(pad_len + int(i))
                if tok_idxs:
                    idx_lists_per_option[oi][b] = torch.tensor(tok_idxs, device=input_ids.device, dtype=torch.long)
                    counts[b][oi] = int(len(tok_idxs))

        debug["option_locator"] = "offset_mapping"
        debug["option_token_counts"] = counts
        return idx_lists_per_option, debug

    @torch.no_grad()
    def _greedy_generate_with_attn_capture(
        self,
        inputs,
        generate_kwargs: dict,
        question_texts: Optional[List[str]] = None,
        prompt_texts: Optional[List[str]] = None,
        option_texts_per_sample: Optional[List[List[str]]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[Optional[torch.Tensor]]], dict]:
        """Greedy decode with low-memory attention capture.

        Runs a prefill forward WITHOUT attentions to build KV-cache, then decodes token-by-token.
        For each decoded token, runs a 1-token forward with output_attentions=True and accumulates
        attention-to-vision vectors.
        """

        debug: dict = {
            "has_attentions": False,
            "attn_steps": 0,
            "attn_layers": None,
        }

        # Locate vision key positions in the prompt
        vis_idx_lists, vis_debug = self._locate_vision_token_indices(inputs.input_ids)
        debug.update(vis_debug)

        # Locate question tokens (optional)
        q_idx_lists = None
        if question_texts is not None:
            q_idx_lists, q_debug = self._locate_question_token_indices_from_prompt(inputs.input_ids, question_texts, prompt_texts)
            debug.update(q_debug)

        # Locate option tokens (optional)
        opt_idx_lists_per_option = None
        if option_texts_per_sample is not None:
            opt_idx_lists_per_option, opt_debug = self._locate_option_token_indices_from_prompt(
                inputs.input_ids,
                prompt_texts,
                option_texts_per_sample,
                max_options=4,
            )
            debug.update(opt_debug)

        max_new_tokens = int(generate_kwargs.get("max_new_tokens", 16))
        eos_token_id = generate_kwargs.get("eos_token_id", self.tokenizer.eos_token_id)

        # Prefill (no attentions) to get cache + logits for first token
        prefill = self.model(
            **inputs,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past = getattr(prefill, "past_key_values", None)
        logits = prefill.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)

        generated = [next_token]
        attn_sum_vis = None
        attn_sum_q = None
        attn_sum_opts: Optional[List[Optional[torch.Tensor]]] = None
        attn_steps = 0

        # Decode remaining tokens; capture attentions for each generated token via 1-token forward.
        for _ in range(max_new_tokens - 1):
            out = self.model(
                input_ids=next_token.unsqueeze(-1),
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past = getattr(out, "past_key_values", past)
            step_attn = getattr(out, "attentions", None)
            if step_attn is not None:
                debug["has_attentions"] = True
                debug["attn_steps"] = debug.get("attn_steps", 0) + 1
                step_vec, step_dbg = self._avg_attn_vec_from_step(step_attn, vis_idx_lists)
                debug["none_layer_count_last"] = step_dbg.get("none_layer_count")
                debug["no_valid_attention_tensors_last"] = step_dbg.get("no_valid_attention_tensors")
                if debug.get("attn_layers") is None:
                    debug["attn_layers"] = step_dbg.get("attn_layers")
                step_vec_q = None
                if q_idx_lists is not None:
                    step_vec_q, _ = self._avg_attn_vec_from_step(step_attn, q_idx_lists)

                step_vec_opts: Optional[List[Optional[torch.Tensor]]] = None
                if opt_idx_lists_per_option is not None:
                    step_vec_opts = []
                    for idx_lists_opt in opt_idx_lists_per_option:
                        vopt, _ = self._avg_attn_vec_from_step(step_attn, idx_lists_opt)
                        step_vec_opts.append(vopt)

                if step_vec is not None or step_vec_q is not None or (step_vec_opts is not None and any(v is not None for v in step_vec_opts)):
                    if step_vec is not None:
                        attn_sum_vis = step_vec if attn_sum_vis is None else (attn_sum_vis + step_vec)
                    if step_vec_q is not None:
                        attn_sum_q = step_vec_q if attn_sum_q is None else (attn_sum_q + step_vec_q)
                    if step_vec_opts is not None:
                        if attn_sum_opts is None:
                            attn_sum_opts = [None for _ in range(len(step_vec_opts))]
                        for iopt, vopt in enumerate(step_vec_opts):
                            if vopt is None:
                                continue
                            attn_sum_opts[iopt] = vopt if attn_sum_opts[iopt] is None else (attn_sum_opts[iopt] + vopt)
                    attn_steps += 1

            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token)

            if eos_token_id is not None:
                eos = eos_token_id
                if isinstance(eos, (list, tuple)):
                    eos_set = set(int(x) for x in eos)
                    if all(int(t.item()) in eos_set for t in next_token):
                        break
                else:
                    if torch.all(next_token.eq(int(eos))):
                        break

        gen_tokens = torch.stack(generated, dim=1)  # (B, Tgen)
        sequences = torch.cat([inputs.input_ids, gen_tokens], dim=1)

        avg_attn_vis = None
        avg_attn_q = None
        avg_attn_opts: Optional[List[Optional[torch.Tensor]]] = None
        if attn_steps > 0:
            if attn_sum_vis is not None:
                avg_attn_vis = attn_sum_vis / float(attn_steps)
            if attn_sum_q is not None:
                avg_attn_q = attn_sum_q / float(attn_steps)
            if attn_sum_opts is not None:
                avg_attn_opts = []
                for vopt in attn_sum_opts:
                    avg_attn_opts.append((vopt / float(attn_steps)) if vopt is not None else None)
        debug["attn_steps_used"] = attn_steps
        return sequences, avg_attn_vis, avg_attn_q, avg_attn_opts, debug

    def _extract_avg_attention_to_vision_tokens(
        self,
        generate_output,
        input_ids: torch.Tensor,
        sequences: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], dict]:
        """Return (batch, n_vision_tokens) averaged over output tokens and decoder layers.

        Uses decoder self-attention weights from generated tokens (query) to
        vision placeholder tokens in the prompt (keys).
        """

        debug: dict = {
            "has_attentions": False,
            "vision_locator": None,
            "vision_token_counts": None,
            "attn_steps": None,
            "attn_layers": None,
        }

        attentions = getattr(generate_output, "attentions", None)
        if attentions is None:
            return None, debug

        debug["has_attentions"] = True

        config = getattr(self.model, "config", None)
        image_token_id = getattr(config, "image_token_id", None)
        video_token_id = getattr(config, "video_token_id", None)
        vision_start_token_id = getattr(config, "vision_start_token_id", None)
        vision_end_token_id = getattr(config, "vision_end_token_id", None)

        idx_lists: list[torch.Tensor] = []
        # 1) Prefer explicit image/video placeholder IDs
        if image_token_id is not None or video_token_id is not None:
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
            if image_token_id is not None:
                mask |= input_ids.eq(int(image_token_id))
            if video_token_id is not None:
                mask |= input_ids.eq(int(video_token_id))
            if mask.any():
                debug["vision_locator"] = "image_token_id/video_token_id"
                for b in range(input_ids.shape[0]):
                    idx_lists.append(torch.nonzero(mask[b], as_tuple=False).squeeze(-1))
        # 2) Fallback: tokens between vision_start and vision_end
        if not idx_lists and vision_start_token_id is not None and vision_end_token_id is not None:
            debug["vision_locator"] = "vision_start->vision_end"
            vs = int(vision_start_token_id)
            ve = int(vision_end_token_id)
            for b in range(input_ids.shape[0]):
                ids = input_ids[b]
                starts = torch.nonzero(ids.eq(vs), as_tuple=False).squeeze(-1)
                ends = torch.nonzero(ids.eq(ve), as_tuple=False).squeeze(-1)
                seg_idxs = []
                if starts.numel() and ends.numel():
                    # Pair each start with the first subsequent end
                    for s in starts.tolist():
                        e_candidates = ends[ends > s]
                        if e_candidates.numel() == 0:
                            continue
                        e = int(e_candidates[0].item())
                        if e > s + 1:
                            seg_idxs.append(torch.arange(s + 1, e, device=ids.device))
                if seg_idxs:
                    idx_lists.append(torch.cat(seg_idxs))
                else:
                    idx_lists.append(ids.new_zeros((0,), dtype=torch.long))

        if not idx_lists:
            debug["vision_locator"] = "none"
            return None, debug

        # Record vision token counts early so we can debug attention issues separately.
        debug["vision_token_counts"] = [int(x.numel()) for x in idx_lists]

        def _as_steps(attn_obj):
            # Common formats:
            # 1) tuple(steps) of tuple(layers) of Tensor
            # 2) tuple(layers) of Tensor (full attention matrices)
            if isinstance(attn_obj, (tuple, list)) and attn_obj:
                first = attn_obj[0]
                if torch.is_tensor(first):
                    return [attn_obj]
            return list(attn_obj) if isinstance(attn_obj, (tuple, list)) else [attn_obj]

        steps = _as_steps(attentions)

        # attentions steps: each step is iterable(layers) with tensors shaped like
        # (B, heads, tgt_len, src_len) or sometimes (B, heads, src_len) for tgt_len==1.
        step_avgs: list[torch.Tensor] = []
        debug["attn_steps"] = len(steps)
        for step_attn in steps:
            if step_attn is None:
                continue
            layer_tensors = []
            for layer_attn in (step_attn if isinstance(step_attn, (tuple, list)) else [step_attn]):
                if layer_attn is None:
                    continue
                # Normalize to (B, heads, tgt_len, src_len)
                if layer_attn.ndim == 3:
                    layer_attn = layer_attn.unsqueeze(-2)
                if layer_attn.ndim != 4:
                    continue
                layer_tensors.append(layer_attn)
            if not layer_tensors:
                continue

            if debug["attn_layers"] is None:
                debug["attn_layers"] = len(layer_tensors)

            layers = torch.stack(layer_tensors, dim=0)  # (L, B, H, T, S)

            # Choose which query tokens to average over.
            # - For step-wise attentions: typically T==1, use -1.
            # - For full-matrix attentions: T can be total sequence length; average
            #   over generated positions [prompt_len:total_len).
            prompt_len = int(input_ids.shape[1])
            total_len = int(sequences.shape[1]) if sequences is not None and sequences.ndim == 2 else None
            tgt_len = int(layers.shape[-2])
            if total_len is not None and tgt_len == total_len and total_len > prompt_len:
                q = layers[:, :, :, prompt_len:total_len, :].mean(dim=-2)  # (L, B, H, S)
            else:
                q = layers[:, :, :, -1, :]  # (L, B, H, S)

            q = q.mean(dim=0)  # avg layers -> (B, H, S)
            q = q.mean(dim=1)  # avg heads  -> (B, S)
            step_avgs.append(q)

        if not step_avgs:
            debug["no_valid_attention_tensors"] = True
            return None, debug

        avg_over_steps = torch.stack(step_avgs, dim=0).mean(dim=0)  # (B, S)

        # Gather only vision placeholder tokens and pad to max
        max_n = max(int(x.numel()) for x in idx_lists)
        out = avg_over_steps.new_zeros((input_ids.shape[0], max_n))
        for b, idx in enumerate(idx_lists):
            if idx.numel() == 0:
                continue
            out[b, : idx.numel()] = avg_over_steps[b, idx]
        debug["vision_token_counts"] = [int(x.numel()) for x in idx_lists]
        return out, debug

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))

        with ThreadPoolExecutor(max_workers=1) as executor:
            save_attn_env = os.environ.get("LMMS_EVAL_EXPERIMENTS_SAVE_ATTN", "0")
            save_attn_default = save_attn_env not in ("", "0", "false", "False")
            future = executor.submit(self._preprocess_chunk, chunks[0], save_attn_default) if chunks else None

            for idx in range(len(chunks)):
                if save_attn_default:
                    inputs, contexts, gen_kwargs, until, rendered_prompts, image_inputs, video_inputs, video_metadata_list = future.result()
                else:
                    inputs, contexts, gen_kwargs, until = future.result()
                    rendered_prompts, image_inputs, video_inputs, video_metadata_list = None, None, None, None

                if idx + 1 < len(chunks):
                    future = executor.submit(self._preprocess_chunk, chunks[idx + 1], save_attn_default)

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)

                # Optional per-request override
                save_attn = bool(gen_kwargs.pop("save_attention", save_attn_default))
                save_dir = gen_kwargs.pop("save_attention_dir", None) or os.environ.get(
                    "LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR",
                    "./experiment_artifacts/qwen3_vl_experiments",
                )

                save_mp4_env = os.environ.get("LMMS_EVAL_EXPERIMENTS_SAVE_MP4", "0")
                save_mp4_default = save_mp4_env not in ("", "0", "false", "False")
                save_mp4 = bool(gen_kwargs.pop("save_mp4", save_mp4_default))

                save_npz_env = os.environ.get("LMMS_EVAL_EXPERIMENTS_SAVE_NPZ", "0")
                save_npz_default = save_npz_env not in ("", "0", "false", "False")
                save_npz = bool(gen_kwargs.pop("save_npz", save_npz_default))

                generate_kwargs = self._build_generate_kwargs(gen_kwargs)
                avg_attn_to_vision = None
                avg_attn_to_question = None
                avg_attn_to_options = None
                attn_debug = {}
                if save_attn:
                    # Recover docs early to get question strings.
                    _, _, _, doc_ids, tasks, splits = zip(*chunks[idx])
                    question_texts = []
                    option_texts_per_sample = []
                    for b in range(len(doc_ids)):
                        try:
                            doc = self.task_dict[tasks[b]][splits[b]][doc_ids[b]]
                            question_texts.append(doc.get("question", "") if isinstance(doc, dict) else "")
                        except Exception:
                            question_texts.append("")
                        # Prefer extracting options from the actually-rendered prompt
                        if rendered_prompts is not None and b < len(rendered_prompts):
                            option_texts_per_sample.append(_extract_options_from_prompt(rendered_prompts[b]))
                        else:
                            try:
                                doc = self.task_dict[tasks[b]][splits[b]][doc_ids[b]]
                                option_texts_per_sample.append(list(doc.get("options", []) or []) if isinstance(doc, dict) else [])
                            except Exception:
                                option_texts_per_sample.append([])
                    # Low-memory path: no attentions during prefill; per-step attentions only.
                    sequences, avg_attn_to_vision, avg_attn_to_question, avg_attn_to_options, attn_debug = self._greedy_generate_with_attn_capture(
                        inputs,
                        generate_kwargs,
                        question_texts=question_texts,
                        prompt_texts=rendered_prompts,
                        option_texts_per_sample=option_texts_per_sample,
                    )
                else:
                    cont = self.model.generate(**inputs, **generate_kwargs)
                    sequences = cont.sequences if hasattr(cont, "sequences") else cont

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, sequences)]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    ans = self._strip_thinking(ans)
                    res.append(ans)
                    self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                    pbar.update(1)

                if save_attn:
                    out_dir = Path(save_dir)
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    batch_n = inputs.input_ids.shape[0]
                    # Recover task/split/doc_id for per-sample metadata
                    _, _, _, doc_ids, tasks, splits = zip(*chunks[idx])
                    for b in range(batch_n):
                        task_name = tasks[b] if b < len(tasks) else None
                        split_name = splits[b] if b < len(splits) else None
                        doc_id = doc_ids[b] if b < len(doc_ids) else None
                        doc = None
                        if task_name is not None and split_name is not None and doc_id is not None:
                            try:
                                doc = self.task_dict[task_name][split_name][doc_id]
                            except Exception:
                                doc = None

                        attn_vec = None
                        if avg_attn_to_vision is not None:
                            attn_vec = avg_attn_to_vision[b].detach().float().cpu().numpy()
                        q_attn_vec = None
                        if avg_attn_to_question is not None:
                            q_attn_vec = avg_attn_to_question[b].detach().float().cpu().numpy()

                        opt_attn_vecs = None
                        if avg_attn_to_options is not None:
                            opt_attn_vecs = []
                            for vopt in avg_attn_to_options:
                                if vopt is None:
                                    opt_attn_vecs.append(None)
                                else:
                                    opt_attn_vecs.append(vopt[b].detach().float().cpu().numpy())
                        # Determine frame count
                        video_frames = None
                        fps = None
                        if video_inputs is not None and b < len(video_inputs):
                            video_frames = _to_uint8_video(video_inputs[b])
                        if video_frames is None and image_inputs is not None and b < len(image_inputs):
                            video_frames = _images_to_uint8_video(image_inputs[b])
                        if video_frames is None:
                            video_frames = _doc_images_to_uint8_video(doc)
                        if video_metadata_list is not None and b < len(video_metadata_list):
                            md = video_metadata_list[b] or {}
                            fps = md.get("fps") if isinstance(md, dict) else None
                        num_frames = int(video_frames.shape[0]) if isinstance(video_frames, np.ndarray) and video_frames.ndim == 4 else 1

                        attn_map_thw = None
                        if attn_vec is not None:
                            # Prefer model-provided grid metadata when available.
                            grid_src = None
                            grid_thw = None
                            grid_merge = 1
                            try:
                                vision_cfg = getattr(getattr(self.model, "config", None), "vision_config", None)
                                grid_merge = int(getattr(vision_cfg, "spatial_merge_size", 1)) if vision_cfg is not None else 1
                            except Exception:
                                grid_merge = 1

                            image_grid_thw = None
                            video_grid_thw = None
                            try:
                                image_grid_thw = inputs.get("image_grid_thw")
                            except Exception:
                                image_grid_thw = getattr(inputs, "image_grid_thw", None)
                            try:
                                video_grid_thw = inputs.get("video_grid_thw")
                            except Exception:
                                video_grid_thw = getattr(inputs, "video_grid_thw", None)

                            if image_grid_thw is not None and torch.is_tensor(image_grid_thw):
                                if image_grid_thw.ndim == 2 and int(image_grid_thw.shape[0]) == int(batch_n):
                                    grid_src = "image_grid_thw"
                                    grid_thw = image_grid_thw[b].detach().cpu().tolist()
                            if grid_thw is None and video_grid_thw is not None and torch.is_tensor(video_grid_thw):
                                if video_grid_thw.ndim == 2 and int(video_grid_thw.shape[0]) == int(batch_n):
                                    grid_src = "video_grid_thw"
                                    grid_thw = video_grid_thw[b].detach().cpu().tolist()

                            if grid_thw is not None and len(grid_thw) == 3:
                                gt, gh, gw = [int(x) for x in grid_thw]
                                ghm = max(1, gh // max(1, grid_merge))
                                gwm = max(1, gw // max(1, grid_merge))
                                expected = max(1, gt) * ghm * gwm
                                if expected > 0 and int(attn_vec.size) == int(expected):
                                    attn_map = attn_vec.reshape(max(1, gt), ghm, gwm)
                                    attn_map_thw = attn_map
                                else:
                                    grid_src = None

                            if grid_src is None:
                                # Split attention vector into per-frame tokens if possible (heuristic).
                                if num_frames > 0 and attn_vec.size % num_frames == 0:
                                    tokens_per_frame = attn_vec.size // num_frames
                                else:
                                    num_frames = 1
                                    tokens_per_frame = attn_vec.size

                                h, w = _best_factor_pair(tokens_per_frame)
                                attn_map = attn_vec[: num_frames * tokens_per_frame].reshape(num_frames, tokens_per_frame)
                                attn_map_thw = attn_map.reshape(num_frames, h, w)

                            if isinstance(attn_debug, dict):
                                attn_debug["attn_grid_source"] = grid_src or "heuristic"
                                attn_debug["attn_grid_thw"] = grid_thw
                                attn_debug["attn_grid_merge"] = int(grid_merge)

                        prompt_text = None
                        if rendered_prompts is not None and b < len(rendered_prompts):
                            prompt_text = rendered_prompts[b]

                        vsibench_score = None
                        vsibench_score_debug = None
                        if isinstance(doc, dict):
                            vsibench_score, vsibench_score_debug = _compute_vsibench_accuracy(doc, answers[b] if b < len(answers) else "")

                        task_type = doc.get("question_type") if isinstance(doc, dict) else None

                        question_text = doc.get("question") if isinstance(doc, dict) else None
                        question_words = None
                        question_word_attn = None
                        if isinstance(question_text, str) and q_attn_vec is not None and q_attn_vec.size > 0:
                            # Prefer offset-mapping aggregation (robust to leading-space/BPE tokenization)
                            prompt = prompt_text or ""
                            q_find = question_text
                            start_char = prompt.find(q_find) if prompt and q_find else -1
                            if start_char < 0 and prompt:
                                q_find = question_text.strip()
                                start_char = prompt.find(q_find)
                            end_char = start_char + len(q_find) if start_char >= 0 else -1

                            token_attn = q_attn_vec.astype(float).tolist()
                            q_n = None
                            try:
                                if isinstance(attn_debug.get("question_token_counts"), list) and b < len(attn_debug["question_token_counts"]):
                                    q_n = int(attn_debug["question_token_counts"][b])
                            except Exception:
                                q_n = None
                            if q_n is not None and q_n > 0:
                                token_attn = token_attn[:q_n]

                            tok_idxs_prompt: list[int] = []
                            if start_char >= 0 and prompt:
                                try:
                                    enc = self.tokenizer(
                                        prompt,
                                        add_special_tokens=False,
                                        return_offsets_mapping=True,
                                    )
                                    offsets = enc.get("offset_mapping")
                                    if offsets is not None:
                                        for i, (s, e) in enumerate(offsets):
                                            if s is None or e is None or e <= s:
                                                continue
                                            if s < end_char and e > start_char:
                                                tok_idxs_prompt.append(int(i))
                                except Exception:
                                    tok_idxs_prompt = []

                            if tok_idxs_prompt and len(tok_idxs_prompt) == len(token_attn) and start_char >= 0:
                                question_words, question_word_attn = _aggregate_question_attn_to_words_from_offsets(
                                    tokenizer=self.tokenizer,
                                    prompt_text=prompt,
                                    question_text=question_text,
                                    question_char_span=(int(start_char), int(end_char)),
                                    question_token_indices_prompt=tok_idxs_prompt,
                                    token_attn=[float(x) for x in token_attn],
                                )
                            else:
                                # Fallback: old heuristic mapping
                                try:
                                    q_idx_lists2, _ = self._locate_question_token_indices_from_prompt(
                                        inputs.input_ids[b : b + 1],
                                        [question_text],
                                        [prompt] if prompt else None,
                                    )
                                    q_idx = q_idx_lists2[0]
                                    q_token_ids = [int(x) for x in inputs.input_ids[b, q_idx].detach().cpu().tolist()] if q_idx.numel() else []
                                except Exception:
                                    q_token_ids = []

                                if q_token_ids:
                                    if len(token_attn) < len(q_token_ids):
                                        token_attn = token_attn + [0.0] * (len(q_token_ids) - len(token_attn))
                                    token_attn = token_attn[: len(q_token_ids)]
                                else:
                                    token_attn = []

                                question_words, question_word_attn = _aggregate_token_attn_to_words(
                                    self.tokenizer,
                                    question_text,
                                    q_token_ids,
                                    token_attn,
                                )

                        # Option word attention (for MCA prompts)
                        options_texts = _extract_options_from_prompt(prompt_text) if isinstance(prompt_text, str) else []
                        option_words = None
                        option_word_attn = None
                        if options_texts and opt_attn_vecs is not None:
                            option_words = []
                            option_word_attn = []
                            for oi, opt_text in enumerate(options_texts[: len(opt_attn_vecs)]):
                                attn_vec = opt_attn_vecs[oi]
                                if attn_vec is None:
                                    option_words.append(_split_words(opt_text))
                                    option_word_attn.append([0.0 for _ in option_words[-1]])
                                    continue

                                # Slice to the true token count if available
                                try:
                                    n_tok = None
                                    if isinstance(attn_debug.get("option_token_counts"), list) and b < len(attn_debug["option_token_counts"]):
                                        cnts = attn_debug["option_token_counts"][b]
                                        if isinstance(cnts, list) and oi < len(cnts):
                                            n_tok = int(cnts[oi])
                                    token_attn = attn_vec.astype(float).tolist()
                                    if n_tok is not None and n_tok > 0:
                                        token_attn = token_attn[:n_tok]
                                except Exception:
                                    token_attn = attn_vec.astype(float).tolist()

                                prompt = prompt_text or ""
                                start_char = prompt.find(opt_text) if prompt else -1
                                end_char = start_char + len(opt_text) if start_char >= 0 else -1

                                tok_idxs_prompt: list[int] = []
                                if start_char >= 0 and prompt:
                                    try:
                                        enc = self.tokenizer(
                                            prompt,
                                            add_special_tokens=False,
                                            return_offsets_mapping=True,
                                        )
                                        offsets = enc.get("offset_mapping")
                                        if offsets is not None:
                                            for i, (s, e) in enumerate(offsets):
                                                if s is None or e is None or e <= s:
                                                    continue
                                                if s < end_char and e > start_char:
                                                    tok_idxs_prompt.append(int(i))
                                    except Exception:
                                        tok_idxs_prompt = []

                                if tok_idxs_prompt and len(tok_idxs_prompt) == len(token_attn) and start_char >= 0:
                                    w, wa = _aggregate_question_attn_to_words_from_offsets(
                                        tokenizer=self.tokenizer,
                                        prompt_text=prompt,
                                        question_text=opt_text,
                                        question_char_span=(int(start_char), int(end_char)),
                                        question_token_indices_prompt=tok_idxs_prompt,
                                        token_attn=[float(x) for x in token_attn],
                                    )
                                else:
                                    w = _split_words(opt_text)
                                    wa = [0.0 for _ in w]
                                option_words.append(w)
                                option_word_attn.append(wa)

                        sample_tag = f"{now}_b{b}_chunk{idx}"
                        metadata = {
                            "model": "qwen3_vl_experiments",
                            "timestamp": now,
                            "chunk_index": idx,
                            "batch_index": b,
                            "task": task_name,
                            "split": split_name,
                            "doc_id": doc_id,
                            "prompt_rendered": prompt_text,
                            "prompt_context": contexts[b] if b < len(contexts) else None,
                            "answer": answers[b] if b < len(answers) else None,
                            "ground_truth": doc.get("ground_truth") if isinstance(doc, dict) and doc.get("ground_truth") is not None else doc.get("Answer") if isinstance(doc, dict) else None,
                            "question_type": doc.get("question_type") if isinstance(doc, dict) else None,
                            "question_text": question_text,
                            "question_words": question_words,
                            "question_word_attn": question_word_attn,
                            "options_texts": options_texts if options_texts else None,
                            "option_words": option_words,
                            "option_word_attn": option_word_attn,
                            "vsibench_accuracy": vsibench_score,
                            "vsibench_accuracy_debug": vsibench_score_debug,
                            "gen_kwargs": gen_kwargs,
                            "video_fps": fps,
                            "attn_debug": attn_debug,
                            "attn_available": attn_map_thw is not None,
                            "attn_frames": int(attn_map_thw.shape[0]) if attn_map_thw is not None else None,
                            "attn_h": int(attn_map_thw.shape[1]) if attn_map_thw is not None else None,
                            "attn_w": int(attn_map_thw.shape[2]) if attn_map_thw is not None else None,
                        }
                        task_save = False

                        # Also save split by task type (VSIBench question_type)
                        if task_type:
                            out_dir_by_type = out_dir / "by_task_type" / _safe_path_component(str(task_type))
                            is_mcq = False
                            try:
                                from lmms_eval.tasks.vsibench.utils import MCA_QUESTION_TYPES

                                is_mcq = str(task_type) in MCA_QUESTION_TYPES
                            except Exception:
                                is_mcq = False

                            if not is_mcq or vsibench_score is None:
                                _save_experiment_artifacts(
                                    out_dir_by_type,
                                    sample_tag,
                                    attn_map_thw,
                                    video_frames,
                                    metadata,
                                    save_mp4=save_mp4,
                                    save_npz=save_npz,
                                )
                            task_save = True

                            if is_mcq and vsibench_score is not None:
                                mcq_bucket = "correct" if float(vsibench_score) >= 1.0 else "incorrect"
                                out_dir_by_mcq = out_dir_by_type / mcq_bucket
                                _save_experiment_artifacts(
                                    out_dir_by_mcq,
                                    sample_tag,
                                    attn_map_thw,
                                    video_frames,
                                    metadata,
                                    save_mp4=save_mp4,
                                    save_npz=save_npz,
                                )

                        if not task_save:
                            # Save default layout
                            _save_experiment_artifacts(
                                out_dir,
                                sample_tag,
                                attn_map_thw,
                                video_frames,
                                metadata,
                                save_mp4=save_mp4,
                                save_npz=save_npz,
                            )

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
