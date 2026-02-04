import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LANGUAGE,
    MODEL_BASE,
    REF_MAX_DURATION_SEC,
    TTS_MAX_NEW_TOKENS,
    TTS_VARY_PROSODY,
    TTS_TEMPERATURE_MIN,
    TTS_TEMPERATURE_MAX,
    TTS_TOP_P_MIN,
    TTS_TOP_P_MAX,
    TTS_SUBTALKER_TEMP_MIN,
    TTS_SUBTALKER_TEMP_MAX,
    TTS_SUBTALKER_TOP_P_MIN,
    TTS_SUBTALKER_TOP_P_MAX,
    TTS_SUBTALKER_TOP_K_MIN,
    TTS_SUBTALKER_TOP_K_MAX,
    TTS_REPETITION_PENALTY_MIN,
    TTS_REPETITION_PENALTY_MAX,
    TTS_BALANCE_REFS,
)
from .denoise import load_and_denoise_wav
from .manifest import write_manifest, write_rejected_manifest
from .quality import filter_wav_quality
from .qwen_backend import Qwen3TTSBackend
from .text_loader import load_texts
from .writer import save_batch_wavs


def _ensure_ref_audio_max_duration(
    ref_audio: Union[str, Tuple[np.ndarray, int]],
    max_sec: float = REF_MAX_DURATION_SEC,
) -> Union[str, Tuple[np.ndarray, int]]:
    """
    참조 음성이 3초 넘어도 오류 나지 않도록.
    """
    if isinstance(ref_audio, str):
        try:
            import librosa
            wav, sr = librosa.load(ref_audio, sr=None, mono=True, duration=max_sec)
            return (wav.astype(np.float32), int(sr))
        except Exception:
            return ref_audio
    wav, sr = ref_audio
    if wav is None or sr is None or len(wav) == 0:
        return ref_audio
    max_samples = int(sr * max_sec)
    if len(wav) <= max_samples:
        return ref_audio
    return (wav[:max_samples].astype(np.float32), int(sr))


def _normalize_ref_audio(
    ref_audio: Union[str, List[str]],
    denoise_ref: bool,
    max_sec: float,
) -> List[Union[str, Tuple[np.ndarray, int]]]:
    """참조 음성 경로를 리스트로 정규화하고, 로드·노이즈제거·길이 보정."""
    if isinstance(ref_audio, str):
        ref_audio = [p.strip() for p in ref_audio.split(",") if p.strip()]
    if not ref_audio:
        raise ValueError("참조 음성이 비어 있습니다.")
    result: List[Union[str, Tuple[np.ndarray, int]]] = []
    for path in ref_audio:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"참조 음성 파일 없음: {path}")
        inp: Union[str, tuple] = path
        if denoise_ref:
            try:
                inp = load_and_denoise_wav(path)
            except RuntimeError as e:
                import warnings
                warnings.warn(f"노이즈 제거 건너뜀 ({path}): {e}", stacklevel=1)
        inp = _ensure_ref_audio_max_duration(inp, max_sec=max_sec)
        result.append(inp)
    return result


def _sample_gen_kwargs() -> Dict[str, Any]:
    """말빠르기·억양·어조 다양화용 샘플링 파라미터 (배치마다 다르게)."""
    if not TTS_VARY_PROSODY:
        return {}
    return {
        "temperature": random.uniform(TTS_TEMPERATURE_MIN, TTS_TEMPERATURE_MAX),
        "top_p": random.uniform(TTS_TOP_P_MIN, TTS_TOP_P_MAX),
        "repetition_penalty": random.uniform(
            TTS_REPETITION_PENALTY_MIN, TTS_REPETITION_PENALTY_MAX
        ),
        "subtalker_temperature": random.uniform(
            TTS_SUBTALKER_TEMP_MIN, TTS_SUBTALKER_TEMP_MAX
        ),
        "subtalker_top_p": random.uniform(
            TTS_SUBTALKER_TOP_P_MIN, TTS_SUBTALKER_TOP_P_MAX
        ),
        "subtalker_top_k": random.randint(
            TTS_SUBTALKER_TOP_K_MIN, TTS_SUBTALKER_TOP_K_MAX
        ),
    }


def run_voice_clone(
    text_file: str,
    output_dir: str,
    ref_audio: Union[str, List[str]],
    ref_text: str = "",
    language: str = DEFAULT_LANGUAGE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    x_vector_only_mode: bool = False,
    model_name: Optional[str] = None,
    write_manifest_csv: bool = True,
    filter_quality: bool = False,
    quality_kwargs: Optional[Dict[str, float]] = None,
    denoise_ref: bool = True,
) -> None:
   
    texts = load_texts(text_file)
    if not texts:
        raise ValueError(f"텍스트가 비어 있습니다: {text_file}")

    ref_inputs = _normalize_ref_audio(ref_audio, denoise_ref, REF_MAX_DURATION_SEC)
    name = model_name or MODEL_BASE
    backend = Qwen3TTSBackend(model_name=name)

    # 단일 참조: 기존처럼 한 프롬프트로 전체 생성
    # 복수 참조: 문장마다 랜덤 참조 → 어조/성별 등 다양
    if len(ref_inputs) == 1:
        prompt: Any = backend.create_voice_clone_prompt(
            ref_audio=ref_inputs[0],
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        all_prompt_items: Optional[List[Any]] = None
    else:
        # 참조별로 프롬프트 아이템 1개씩 생성
        multi_prompt = backend.create_voice_clone_prompt(
            ref_audio=ref_inputs,
            ref_text=[ref_text] * len(ref_inputs),
            x_vector_only_mode=[x_vector_only_mode] * len(ref_inputs),
        )
        n_refs = len(multi_prompt)
        n_texts = len(texts)
        # 남녀 등 50:50 균등 배정 (TTS_BALANCE_REFS=1) 또는 완전 랜덤
        if TTS_BALANCE_REFS and n_refs == 2:
            half = (n_texts + 1) // 2
            ref_indices = [0] * half + [1] * (n_texts - half)
            random.shuffle(ref_indices)
            all_prompt_items = [multi_prompt[ref_indices[i]] for i in range(n_texts)]
        else:
            all_prompt_items = [random.choice(multi_prompt) for _ in texts]
        prompt = None

    all_paths: List[str] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        idx_start = start + 1
        if all_prompt_items is not None:
            batch_prompts = all_prompt_items[start : start + batch_size]
            voice_clone_prompt = batch_prompts
        else:
            voice_clone_prompt = prompt
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": TTS_MAX_NEW_TOKENS,
            **_sample_gen_kwargs(),
        }
        wavs, sr = backend.generate_voice_clone(
            text=batch,
            language=[language] * len(batch),
            voice_clone_prompt=voice_clone_prompt,
            **gen_kwargs,
        )
        paths = save_batch_wavs(
            output_dir,
            wavs,
            sr,
            prefix="voice_clone_",
            start_idx=idx_start,
        )
        all_paths.extend(paths)

    if not write_manifest_csv:
        return

    if filter_quality:
        kwargs = quality_kwargs or {}
        (passed_paths, passed_sentences), (
            rejected_paths,
            rejected_sentences,
            reasons,
        ) = filter_wav_quality(all_paths, texts, **kwargs)
        write_manifest(
            os.path.join(output_dir, "train.csv"),
            passed_paths,
            passed_sentences,
            use_basename=True,
        )
        if rejected_paths:
            write_rejected_manifest(
                os.path.join(output_dir, "rejected.csv"),
                rejected_paths,
                rejected_sentences,
                reasons,
                use_basename=True,
            )
    else:
        write_manifest(
            os.path.join(output_dir, "train.csv"),
            all_paths,
            texts,
            use_basename=True,
        )


def run_filter_quality(
    wav_dir: str,
    text_file: str,
    *,
    prefix: str = "voice_clone_",
    quality_kwargs: Optional[Dict[str, float]] = None,
) -> None:
    """
    이미 생성된 WAV 디렉터리와 문장 파일로 품질 검사 후 train.csv / rejected.csv 생성.
    WAV 파일명은 prefix + 숫자 + .wav (예: voice_clone_0001.wav) 순서로 문장과 1:1 매칭.

    Input:
        wav_dir: WAV가 있는 디렉터리.
        text_file: 한 줄당 한 문장 (WAV 순서와 동일).
        prefix: WAV 파일 prefix (기본 voice_clone_).
        quality_kwargs: 품질 검사 인자. None이면 기본값.
    """
    texts = load_texts(text_file)
    if not texts:
        raise ValueError(f"텍스트가 비어 있습니다: {text_file}")

    pattern = re.compile(re.escape(prefix) + r"(\d+)\.wav$", re.IGNORECASE)
    paths_with_num: List[Tuple[int, str]] = []
    for f in os.listdir(wav_dir):
        m = pattern.match(f)
        if m:
            paths_with_num.append((int(m.group(1)), os.path.join(wav_dir, f)))
    paths_with_num.sort(key=lambda x: x[0])
    wav_paths = [p for _, p in paths_with_num]

    if len(wav_paths) > len(texts):
        wav_paths = wav_paths[: len(texts)]
    elif len(texts) > len(wav_paths):
        texts = texts[: len(wav_paths)]
    if not wav_paths:
        raise ValueError(f"WAV 파일을 찾을 수 없습니다: {wav_dir} (prefix={prefix})")

    kwargs = quality_kwargs or {}
    (passed_paths, passed_sentences), (
        rejected_paths,
        rejected_sentences,
        reasons,
    ) = filter_wav_quality(wav_paths, texts, **kwargs)
    write_manifest(
        os.path.join(wav_dir, "train.csv"),
        passed_paths,
        passed_sentences,
        use_basename=True,
    )
    if rejected_paths:
        write_rejected_manifest(
            os.path.join(wav_dir, "rejected.csv"),
            rejected_paths,
            rejected_sentences,
            reasons,
            use_basename=True,
        )