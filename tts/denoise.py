"""
WAV 노이즈 제거. Voice clone 전에 음성만 정제할 때 사용.
"""

from typing import Tuple

import numpy as np
import soundfile as sf

try:
    import noisereduce as nr
except ImportError:
    nr = None


def load_and_denoise_wav(path: str) -> Tuple[np.ndarray, int]:
    """
    WAV 파일을 읽어 노이즈 제거.
    soundfile 실패 시(포맷 미인식 등) librosa로 로드 후 노이즈 제거.

    Input:
        path: WAV 파일 경로.

    Returns:
        (denoised_audio_float32, sample_rate)

    Raises:
        FileNotFoundError: 파일 없을 때.
        RuntimeError: noisereduce 미설치 시.
    """
    if nr is None:
        raise RuntimeError("노이즈 제거를 쓰려면 noisereduce를 설치하세요: pip install noisereduce")

    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        import librosa
        data, sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        sr = int(sr)

    if data.ndim > 1:
        data = data.mean(axis=1)

    reduced = nr.reduce_noise(y=data, sr=sr, prop_decrease=1.0)
    return reduced.astype(np.float32), int(sr)