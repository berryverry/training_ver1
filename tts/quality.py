"""
WAV 품질 검사: 무음 비율, 클리핑, 길이 등으로 나쁜 데이터 필터링.

Description:
    Voice clone 등으로 생성된 WAV가 finetuning에 적합한지 보고,
    pass, not pass 목록을 반환한다.
"""

from typing import List, Tuple

import numpy as np
import soundfile as sf


def check_wav_quality(
    wav_path: str,
    *,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    silence_threshold: float = 0.01,
    silence_ratio_max: float = 0.9,
    clip_threshold: float = 0.99,
    frame_sec: float = 0.01,
) -> Tuple[bool, str]:
    """
    WAV 파일 품질 검사.
    """
    try:
        data, sr = sf.read(wav_path, dtype="float64")
    except Exception as e:
        return False, f"read_error:{e!s}"

    if data.ndim > 1:
        data = data.mean(axis=1)

    duration = len(data) / sr
    if duration < min_duration:
        return False, "too_short"
    if duration > max_duration:
        return False, "too_long"

    # Clipping
    peak = np.max(np.abs(data))
    if peak >= clip_threshold:
        return False, "clipping"

    # Silence ratio (RMS per frame)
    frame_len = int(sr * frame_sec)
    if frame_len < 1:
        frame_len = 1
    n_frames = (len(data) - frame_len) // frame_len + 1
    if n_frames <= 0:
        return False, "too_short"
    rms_per_frame = np.array(
        [
            np.sqrt(np.mean(data[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ]
    )
    silent_frames = np.sum(rms_per_frame < silence_threshold)
    if silent_frames / n_frames > silence_ratio_max:
        return False, "too_silent"

    return True, ""


def filter_wav_quality(
    wav_paths: List[str],
    sentences: List[str],
    *,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    silence_threshold: float = 0.01,
    silence_ratio_max: float = 0.9,
    clip_threshold: float = 0.99,
) -> Tuple[
    Tuple[List[str], List[str]],
    Tuple[List[str], List[str], List[str]],
]:

    if len(wav_paths) != len(sentences):
        raise ValueError(
            f"wav_paths({len(wav_paths)})와 sentences({len(sentences)}) 길이가 같아야 합니다."
        )
    passed_paths: List[str] = []
    passed_sentences: List[str] = []
    rejected_paths: List[str] = []
    rejected_sentences: List[str] = []
    reasons: List[str] = []

    kwargs = dict(
        min_duration=min_duration,
        max_duration=max_duration,
        silence_threshold=silence_threshold,
        silence_ratio_max=silence_ratio_max,
        clip_threshold=clip_threshold,
    )
    for path, sent in zip(wav_paths, sentences):
        ok, reason = check_wav_quality(path, **kwargs)
        if ok:
            passed_paths.append(path)
            passed_sentences.append(sent)
        else:
            rejected_paths.append(path)
            rejected_sentences.append(sent)
            reasons.append(reason)

    return (passed_paths, passed_sentences), (rejected_paths, rejected_sentences, reasons)