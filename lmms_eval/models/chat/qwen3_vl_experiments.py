import time
import os
from datetime import datetime
from pathlib import Path
from typing import List

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.qwen3_vl_experiments import (
    Qwen3_VL_Experiments as Qwen3_VL_ExperimentsSimple,
    _best_factor_pair,
    _compute_vsibench_accuracy,
    _extract_options_from_prompt,
    _find_subsequence,
    _aggregate_token_attn_to_words,
    _aggregate_question_attn_to_words_from_offsets,
    _split_words,
    _safe_path_component,
    _save_experiment_artifacts,
    _images_to_uint8_video,
    _doc_images_to_uint8_video,
    _collect_doc_extra_artifact_visuals,
    _to_uint8_video,
    _estimate_video_fps,
)
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_experiments_chat")
class Qwen3_VL_Experiments(Qwen3_VL_ExperimentsSimple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        save_attn_env = os.environ.get("LMMS_EVAL_EXPERIMENTS_SAVE_ATTN", "0")
        save_attn_default = save_attn_env not in ("", "0", "false", "False")
        save_dir_default = os.environ.get(
            "LMMS_EVAL_EXPERIMENTS_ATTENTION_DIR",
            "./experiment_artifacts/qwen3_vl_experiments",
        )
        save_mp4_env = os.environ.get("LMMS_EVAL_EXPERIMENTS_SAVE_MP4", "0")
        save_mp4_default = save_mp4_env not in ("", "0", "false", "False")
        save_npz_env = os.environ.get("LMMS_EVAL_EXPERIMENTS_SAVE_NPZ", "0")
        save_npz_default = save_npz_env not in ("", "0", "false", "False")

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)

            chat_messages: List[ChatMessages] = []
            visuals = []
            videos = []
            for idx, (ids, task_name, split_name) in enumerate(zip(doc_id, task, split)):
                messages = doc_to_messages[idx](self.task_dict[task_name][split_name][ids])
                messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
                chat_message = ChatMessages(**{"messages": messages})
                visual, video, _ = chat_message.extract_media()
                visuals.append(visual)
                videos.append(video)
                chat_messages.append(chat_message)

            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = self._build_video_kwargs()
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]

            texts = self._apply_chat_template(batched_messages)

            image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_kwargs = {**video_kwargs, **video_kwargs_qwen}

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)

            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
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
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    return_tensors="pt",
                )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            generate_kwargs = self._build_generate_kwargs(gen_kwargs)

            save_attn = bool(gen_kwargs.pop("save_attention", save_attn_default))
            save_dir = gen_kwargs.pop("save_attention_dir", None) or save_dir_default
            save_mp4 = bool(gen_kwargs.pop("save_mp4", save_mp4_default))
            save_npz = bool(gen_kwargs.pop("save_npz", save_npz_default))

            start_time = time.time()
            avg_attn_to_vision = None
            avg_attn_to_question = None
            avg_attn_to_options = None
            attn_debug = {}
            if save_attn:
                question_texts = []
                option_texts_per_sample = []
                for b in range(len(doc_id)):
                    try:
                        d = self.task_dict[task[b]][split[b]][doc_id[b]]
                        if isinstance(d, dict):
                            question_texts.append(d.get("question") or d.get("Question") or "")
                        else:
                            question_texts.append("")
                    except Exception:
                        question_texts.append("")

                    # Use the actually-rendered prompt to extract options (most reliable)
                    option_texts_per_sample.append(_extract_options_from_prompt(texts[b] if b < len(texts) else ""))

                sequences, avg_attn_to_vision, avg_attn_to_question, avg_attn_to_options, attn_debug = self._greedy_generate_with_attn_capture(
                    inputs,
                    generate_kwargs,
                    question_texts=question_texts,
                    prompt_texts=texts,
                    option_texts_per_sample=option_texts_per_sample,
                )
            else:
                cont = self.model.generate(**inputs, **generate_kwargs)
                sequences = cont.sequences if hasattr(cont, "sequences") else cont
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, sequences)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                ans = self._strip_thinking(ans)
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")

            if save_attn:
                out_dir = Path(save_dir)
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_n = inputs.input_ids.shape[0]
                for b in range(batch_n):
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
                    doc = None
                    try:
                        doc = self.task_dict[task[b]][split[b]][doc_id[b]]
                    except Exception:
                        doc = None

                    video_frames = None
                    fps = None
                    if video_inputs is not None and b < len(video_inputs):
                        video_frames = _to_uint8_video(video_inputs[b])
                    if video_frames is None and image_inputs is not None and b < len(image_inputs):
                        video_frames = _images_to_uint8_video(image_inputs[b])
                    if video_frames is None:
                        video_frames = _doc_images_to_uint8_video(doc)
                    if video_metadatas is not None and b < len(video_metadatas):
                        md = video_metadatas[b] or {}
                        fps = md.get("fps") if isinstance(md, dict) else None
                    if fps is None and isinstance(doc, dict):
                        video_path = None
                        for key in ("video_path", "video", "Video", "path", "filepath"):
                            val = doc.get(key)
                            if isinstance(val, str) and val:
                                video_path = val
                                break
                        if video_path is None and (doc.get("Video") or doc.get("Source")):
                            try:
                                from lmms_eval.tasks.stibench.utils import stibench_doc_to_visual

                                video_path = stibench_doc_to_visual(doc)[0]
                            except Exception:
                                video_path = None
                        fps = _estimate_video_fps(video_path)

                    num_frames = int(video_frames.shape[0]) if video_frames is not None and video_frames.ndim == 4 else 1
                    attn_map_thw = None
                    if attn_vec is not None:
                        if num_frames > 0 and attn_vec.size % num_frames == 0:
                            tokens_per_frame = attn_vec.size // num_frames
                        else:
                            num_frames = 1
                            tokens_per_frame = attn_vec.size

                        h, w = _best_factor_pair(tokens_per_frame)
                        attn_map_thw = attn_vec[: num_frames * tokens_per_frame].reshape(num_frames, h, w)

                    vsibench_score = None
                    vsibench_score_debug = None
                    if isinstance(doc, dict):
                        vsibench_score, vsibench_score_debug = _compute_vsibench_accuracy(doc, answers[b] if b < len(answers) else "")

                    question_text = None
                    if isinstance(doc, dict):
                        question_text = doc.get("question") or doc.get("Question")
                    question_words = None
                    question_word_attn = None
                    if isinstance(question_text, str) and q_attn_vec is not None and q_attn_vec.size > 0:
                        prompt = texts[b] if b < len(texts) else ""
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
                    prompt_text = texts[b] if b < len(texts) else ""
                    options_texts = _extract_options_from_prompt(prompt_text) if isinstance(prompt_text, str) else []
                    option_words = None
                    option_word_attn = None
                    if options_texts and opt_attn_vecs is not None:
                        option_words = []
                        option_word_attn = []
                        for oi, opt_text in enumerate(options_texts[: len(opt_attn_vecs)]):
                            attn_vec = opt_attn_vecs[oi]
                            if attn_vec is None:
                                w = _split_words(opt_text)
                                option_words.append(w)
                                option_word_attn.append([0.0 for _ in w])
                                continue

                            # Slice to true token count if available
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

                            start_char = prompt_text.find(opt_text) if prompt_text else -1
                            end_char = start_char + len(opt_text) if start_char >= 0 else -1

                            tok_idxs_prompt: list[int] = []
                            if start_char >= 0 and prompt_text:
                                try:
                                    enc = self.tokenizer(
                                        prompt_text,
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
                                    prompt_text=prompt_text,
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

                    sample_tag = f"{now}_{task[b]}_{split[b]}_{doc_id[b]}"
                    task_type = None
                    if isinstance(doc, dict):
                        task_type = doc.get("question_type")
                        if not task_type:
                            task_name = (doc.get("Task") or "").strip()
                            task_type = task_name.lower().replace(" ", "_").replace("&", "and") if task_name else None

                    
                    if doc.get("category") is not None:
                        task_type = doc.get("category")
                    if doc.get("qtype") is not None:
                        q_type = doc.get("qtype").split('_')
                        relation = doc.get("relation").split(' ')
                        task_type = '_'.join(q_type + relation)
                    
                    metadata = {
                        "model": "qwen3_vl_experiments_chat",
                        "timestamp": now,
                        "task": task[b],
                        "split": split[b],
                        "doc_id": int(doc_id[b]),
                        "prompt_rendered": texts[b] if b < len(texts) else None,
                        "answer": answers[b] if b < len(answers) else None,
                        "ground_truth": doc.get("ground_truth") if isinstance(doc, dict) and doc.get("ground_truth") is not None else doc.get("Answer") if doc.get("Answer") is not None else doc.get("answer") if isinstance(doc, dict) else None,
                        "question_type": task_type,
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
                    extra_visuals = _collect_doc_extra_artifact_visuals(doc)
                    task_save = False
                    # Also save split by task type (STI/VSI Bench)
                    if task_type:
                        out_dir_by_type = out_dir / "by_task_type" / _safe_path_component(str(task_type))
                        _save_experiment_artifacts(
                            out_dir_by_type,
                            sample_tag,
                            attn_map_thw,
                            video_frames,
                            metadata,
                            save_mp4=save_mp4,
                            save_npz=save_npz,
                            extra_visuals=extra_visuals,
                        )
                        task_save = True

                        if vsibench_score is not None:
                            is_mcq = False
                            try:
                                from lmms_eval.tasks.vsibench.utils import MCA_QUESTION_TYPES

                                is_mcq = str(task_type) in MCA_QUESTION_TYPES
                            except Exception:
                                is_mcq = False

                            if is_mcq:
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
                                    extra_visuals=extra_visuals,
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
                            extra_visuals=extra_visuals,
                        )
                            
            pbar.update(1)

        res = re_ords.get_original(res)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res
