"""
Description:
    환경변수 TTS_MODE=custom_voice | voice_clone | filter_quality, TTS_TEXT_FILE, TTS_OUTPUT_DIR 등으로
    run_custom_voice, run_voice_clone, run_filter_quality 호출. 미설정 시 아래 RUN_* 상수 사용.
"""

import os

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER,
)
from .custom_voice import run_custom_voice
from .voice_clone import run_filter_quality, run_voice_clone


# ---------------------------------------------------------------------------
# 필요 시 수정
# ---------------------------------------------------------------------------
RUN_MODE = os.environ.get("TTS_MODE", "custom_voice")
RUN_TEXT_FILE = os.environ.get("TTS_TEXT_FILE", "scripts/tts_sentences.txt")
RUN_OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "outputs/tts")
RUN_SPEAKER = os.environ.get("TTS_SPEAKER", DEFAULT_SPEAKER)
RUN_LANGUAGE = os.environ.get("TTS_LANGUAGE", DEFAULT_LANGUAGE)
RUN_BATCH_SIZE = int(os.environ.get("TTS_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
RUN_REF_AUDIO = os.environ.get("TTS_REF_AUDIO", "")
RUN_REF_TEXT = os.environ.get("TTS_REF_TEXT", "")
RUN_X_VECTOR_ONLY = os.environ.get("TTS_X_VECTOR_ONLY", "").lower() in ("1", "true", "yes")
RUN_WRITE_MANIFEST = os.environ.get("TTS_WRITE_MANIFEST", "1").lower() in ("1", "true", "yes")
RUN_FILTER_QUALITY = os.environ.get("TTS_FILTER_QUALITY", "").lower() in ("1", "true", "yes")
RUN_DENOISE_REF = os.environ.get("TTS_DENOISE_REF", "1").lower() in ("1", "true", "yes")


def main() -> None:
    if RUN_MODE == "custom_voice":
        run_custom_voice(
            text_file=RUN_TEXT_FILE,
            output_dir=RUN_OUTPUT_DIR,
            speaker=RUN_SPEAKER,
            language=RUN_LANGUAGE,
            batch_size=RUN_BATCH_SIZE,
        )
    elif RUN_MODE == "voice_clone":
        if not RUN_REF_AUDIO:
            raise SystemExit("voice_clone 모드에서는 TTS_REF_AUDIO가 필요합니다.")
        if not RUN_REF_TEXT and not RUN_X_VECTOR_ONLY:
            raise SystemExit("voice_clone 모드에서는 TTS_REF_TEXT 또는 TTS_X_VECTOR_ONLY=1이 필요합니다.")
        run_voice_clone(
            text_file=RUN_TEXT_FILE,
            output_dir=RUN_OUTPUT_DIR,
            ref_audio=RUN_REF_AUDIO,
            ref_text=RUN_REF_TEXT,
            language=RUN_LANGUAGE,
            batch_size=RUN_BATCH_SIZE,
            x_vector_only_mode=RUN_X_VECTOR_ONLY,
            write_manifest_csv=RUN_WRITE_MANIFEST,
            filter_quality=RUN_FILTER_QUALITY,
            denoise_ref=RUN_DENOISE_REF,
        )
    elif RUN_MODE == "filter_quality":
        run_filter_quality(
            wav_dir=RUN_OUTPUT_DIR,
            text_file=RUN_TEXT_FILE,
        )
    else:
        raise SystemExit(
            f"지원하지 않는 TTS_MODE: {RUN_MODE}. custom_voice | voice_clone | filter_quality"
        )


if __name__ == "__main__":
    main()