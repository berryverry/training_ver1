"""
Finetuning용 CSV 생성 (path, sentence).

Description:
    WAV 경로와 문장 리스트를 finetune_whisper.py가 기대하는 CSV 형식으로 저장.
    path는 data_dir 기준 상대 경로로 쓸 수 있도록 basename만 저장 가능.
"""

import csv
import os
from typing import List


def write_manifest(
    csv_path: str,
    wav_paths: List[str],
    sentences: List[str],
    *,
    use_basename: bool = True,
    path_column: str = "path",
    sentence_column: str = "sentence",
) -> None:
    """
    WAV 경로와 문장을 CSV로 저장. finetune_whisper --data_dir + train_csv 형식.
    """
    if len(wav_paths) != len(sentences):
        raise ValueError(
            f"wav_paths({len(wav_paths)})와 sentences({len(sentences)}) 길이가 같아야 합니다."
        )
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([path_column, sentence_column])
        for path, sent in zip(wav_paths, sentences):
            p = os.path.basename(path) if use_basename else path
            w.writerow([p, sent])


def write_rejected_manifest(
    csv_path: str,
    wav_paths: List[str],
    sentences: List[str],
    reasons: List[str],
    *,
    use_basename: bool = True,
) -> None:
    """
    탈락한 샘플 목록을 path, sentence, reason CSV로 저장.
    """
    if len(wav_paths) != len(sentences) or len(sentences) != len(reasons):
        raise ValueError("wav_paths, sentences, reasons 길이가 같아야 합니다.")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "sentence", "reason"])
        for path, sent, reason in zip(wav_paths, sentences, reasons):
            p = os.path.basename(path) if use_basename else path
            w.writerow([p, sent, reason])