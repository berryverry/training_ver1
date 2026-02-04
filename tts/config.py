"""
TTS 파이프라인 기본 상수.

Description:
    Qwen3-TTS CustomVoice/Base 모델명, 기본 스피커·언어·배치 크기.
"""

import os

MODEL_CUSTOM_VOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_SPEAKER = "Sohee"
DEFAULT_LANGUAGE = "Korean"
# 배치 크기: 클수록 빠름(16~24), 작을수록 어조 다양 (4~8). 환경변수 TTS_BATCH_SIZE 로 변경 가능
DEFAULT_BATCH_SIZE = 16
DEFAULT_INSTRUCT = ""

# 참조 음성: (3초 넘어도 오류 안 나도록). 환경변수 TTS_REF_MAX_DURATION_SEC
REF_MAX_DURATION_SEC = float(os.environ.get("TTS_REF_MAX_DURATION_SEC", "10.0"))

# 합성 시 최대 토큰 수. 클수록 긴 문장(3초 이상) 생성 가능. 환경변수 TTS_MAX_NEW_TOKENS
TTS_MAX_NEW_TOKENS = int(os.environ.get("TTS_MAX_NEW_TOKENS", "8192"))

# 어조/분위기 다양화: 배치마다 temperature, top_p 랜덤 (1=켜기, 0=고정)
# 변화를 더 주려면 범위를 넓히거나 TTS_BATCH_SIZE를 줄이기 (예: 4 또는 2)
TTS_VARY_PROSODY = os.environ.get("TTS_VARY_PROSODY", "1").lower() in ("1", "true", "yes")
TTS_TEMPERATURE_MIN = float(os.environ.get("TTS_TEMPERATURE_MIN", "0.62"))
TTS_TEMPERATURE_MAX = float(os.environ.get("TTS_TEMPERATURE_MAX", "1.35"))
TTS_TOP_P_MIN = float(os.environ.get("TTS_TOP_P_MIN", "0.80"))
TTS_TOP_P_MAX = float(os.environ.get("TTS_TOP_P_MAX", "1.0"))
# 서브토커(발음/리듬/억양) 다양화
TTS_SUBTALKER_TEMP_MIN = float(os.environ.get("TTS_SUBTALKER_TEMP_MIN", "0.75"))
TTS_SUBTALKER_TEMP_MAX = float(os.environ.get("TTS_SUBTALKER_TEMP_MAX", "1.15"))
TTS_SUBTALKER_TOP_P_MIN = float(os.environ.get("TTS_SUBTALKER_TOP_P_MIN", "0.88"))
TTS_SUBTALKER_TOP_P_MAX = float(os.environ.get("TTS_SUBTALKER_TOP_P_MAX", "1.0"))
TTS_SUBTALKER_TOP_K_MIN = int(os.environ.get("TTS_SUBTALKER_TOP_K_MIN", "35"))
TTS_SUBTALKER_TOP_K_MAX = int(os.environ.get("TTS_SUBTALKER_TOP_K_MAX", "55"))
# 반복 억제 → 리듬/말빠르기 느낌 다양 (값 높을수록 단조로움 감소)
TTS_REPETITION_PENALTY_MIN = float(os.environ.get("TTS_REPETITION_PENALTY_MIN", "1.02"))
TTS_REPETITION_PENALTY_MAX = float(os.environ.get("TTS_REPETITION_PENALTY_MAX", "1.12"))
# 참조 2개일 때 50:50 균등 배정 (1=켜기, 0=완전 랜덤). 환경변수 TTS_BALANCE_REFS
TTS_BALANCE_REFS = os.environ.get("TTS_BALANCE_REFS", "1").lower() in ("1", "true", "yes")
