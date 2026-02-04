import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import DEFAULT_BATCH_SIZE, DEFAULT_LANGUAGE, MODEL_BASE
from .denoise import load_and_denoise_wav
from .manifest import write_manifest, write_rejected_manifest
from .quality import filter_wav_quality
from .qwen_backend import Qwen3TTSBackend
from .text_loader import load_texts
from .writer import save_batch_wavs


def run_voice_clone(
    text_file: str,
    output_dir: str,
    ref_audio: str,
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

    ref_audio_input: Union[str, tuple]
    if denoise_ref and isinstance(ref_audio, str) and os.path.isfile(ref_audio):
        try:
            ref_audio_input = load_and_denoise_wav(ref_audio)
        except RuntimeError as e:
            import warnings
            warnings.warn(f"노이즈 제거 건너뜀 (noisereduce 미설치 또는 오류): {e}", stacklevel=1)
            ref_audio_input = ref_audio
    else:
        ref_audio_input = ref_audio

    name = model_name or MODEL_BASE
    backend = Qwen3TTSBackend(model_name=name)
    prompt: Any = backend.create_voice_clone_prompt(
        ref_audio=ref_audio_input,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )

    all_paths: List[str] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        idx_start = start + 1
        wavs, sr = backend.generate_voice_clone(
            text=batch,
            language=[language] * len(batch),
            voice_clone_prompt=prompt,
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