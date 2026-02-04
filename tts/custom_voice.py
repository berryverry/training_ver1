from typing import Optional

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INSTRUCT,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER,
    MODEL_CUSTOM_VOICE,
)
from .qwen_backend import Qwen3TTSBackend
from .text_loader import load_texts
from .writer import save_batch_wavs


def run_custom_voice(
    text_file: str,
    output_dir: str,
    speaker: str = DEFAULT_SPEAKER,
    language: str = DEFAULT_LANGUAGE,
    instruct: str = DEFAULT_INSTRUCT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: Optional[str] = None,
) -> None:
    """
    Description:
        텍스트 파일(한 줄 = 한 문장)을 CustomVoice 모델로 WAV로 대량 생성. output_dir에 custom_voice_0001.wav 형식으로 저장.

    Input:
        text_file: 합성할 문장들이 담긴 파일 경로
        output_dir: WAV 저장 디렉터리.
        speaker: CustomVoice 스피커 (기본: Sohee).
        language: 언어 (기본: Korean).
        instruct: 톤/감정 지시 (기본: 빈 문자열).
        batch_size: 한 번에 합성할 문장 수.
        model_name: HuggingFace 모델 ID. None이면 config 기본 CustomVoice 사용.

    Returns:
        None

    Raises:
        FileNotFoundError: text_file 없을 때.
        ValueError: 텍스트가 비어 있을 때.
    """
    texts = load_texts(text_file)
    if not texts:
        raise ValueError(f"텍스트가 비어 있습니다: {text_file}")

    name = model_name or MODEL_CUSTOM_VOICE
    backend = Qwen3TTSBackend(model_name=name)
    n = len(texts)

    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        idx_start = start + 1
        wavs, sr = backend.generate_custom_voice(
            text=batch,
            language=[language] * len(batch),
            speaker=[speaker] * len(batch),
            instruct=[instruct] * len(batch),
        )
        save_batch_wavs(
            output_dir,
            wavs,
            sr,
            prefix="custom_voice_",
            start_idx=idx_start,
        )