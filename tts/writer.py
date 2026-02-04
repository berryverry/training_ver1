import os
from typing import List

import numpy as np
import soundfile as sf


def save_batch_wavs(
    output_dir: str,
    wavs: List[np.ndarray],
    sample_rate: int,
    prefix: str,
    start_idx: int,
) -> List[str]:
    
    os.makedirs(output_dir, exist_ok=True)
    n_total = len(wavs) + start_idx - 1
    pad = len(str(max(n_total, 1)))
    paths: List[str] = []
    for i, wav in enumerate(wavs):
        idx = start_idx + i
        out_path = os.path.join(output_dir, f"{prefix}{idx:0{pad}d}.wav")
        sf.write(out_path, wav, sample_rate)
        paths.append(out_path)
    return paths
