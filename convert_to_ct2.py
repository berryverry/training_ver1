import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """
    Description:
        HF 형식 Whisper 모델 경로를 받아 ct2-transformers-converter(또는 python -m)로
        CTranslate2 모델로 변환하고, tokenizer 등 필요한 파일을 복사한다.

    Input:
        None (argparse로 model, --output_dir, --quantization, --copy_files 처리)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Whisper HF 모델 → CTranslate2 변환 (faster-whisper용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="./whisper-finetuned",
        help="Hugging Face 형식 Whisper 모델 경로 (파인튜닝 출력 폴더 또는 HF 모델 ID)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=None,
        help="CTranslate2 모델 저장 경로 (기본: <model>-ct2)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="float16",
        choices=["float32", "float16", "int8_float16", "int8"],
        help="양자화 방식 (기본: float16)",
    )
    parser.add_argument(
        "--copy_files",
        type=str,
        nargs="+",
        default=["tokenizer.json", "tokenizer_config.json"],
        help="모델 폴더에서 출력 폴더로 복사할 파일 (기본: tokenizer.json tokenizer_config.json)",
    )
    args = parser.parse_args()

    model_path = args.model
    output_dir = args.output_dir or (str(Path(model_path).resolve()) + "-ct2")

    # 로컬 경로면 절대 경로로 정규화
    if not model_path.startswith(("http", "hf://")) and not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    # 로컬 폴더인데 preprocessor_config.json 없으면 processor_config.json 사용
    copy_files = list(args.copy_files)
    if os.path.isdir(model_path):
        if "preprocessor_config.json" in copy_files and not (Path(model_path) / "preprocessor_config.json").exists():
            if (Path(model_path) / "processor_config.json").exists():
                copy_files = [f.replace("preprocessor_config.json", "processor_config.json") for f in copy_files]
        # 실제 존재하는 파일만 복사 목록에
        copy_files = [f for f in copy_files if (Path(model_path) / f).exists()]
        if not copy_files:
            copy_files = ["tokenizer.json"]
            if (Path(model_path) / "tokenizer.json").exists():
                pass
            else:
                copy_files = []

    cmd = [
        "ct2-transformers-converter",
        "--model",
        model_path,
        "--output_dir",
        output_dir,
        "--quantization",
        args.quantization,
    ]
    for f in copy_files:
        cmd.extend(["--copy_files", f])

    print("실행:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        # PATH에 없으면 python -m으로 시도
        cmd_alt = [sys.executable, "-m", "ctranslate2.converters.transformers"] + cmd[1:]
        print("ct2-transformers-converter 미발견, python -m으로 재시도...")
        try:
            subprocess.run(cmd_alt, check=True)
        except FileNotFoundError:
            print("오류: ctranslate2가 설치되어 있지 않습니다.", file=sys.stderr)
            print("  pip install ctranslate2", file=sys.stderr)
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    # processor_config.json → preprocessor_config.json 복사 (faster-whisper 일부 버전 호환)
    src_dir = Path(model_path)
    dst_dir = Path(output_dir)
    if src_dir.is_dir():
        for name in ("processor_config.json", "preprocessor_config.json"):
            src = src_dir / name
            if src.exists() and not (dst_dir / "preprocessor_config.json").exists():
                shutil.copy2(src, dst_dir / "preprocessor_config.json")
                print(f"  복사: {name} -> {output_dir}/preprocessor_config.json")
                break

    print(f"\n변환 완료: {output_dir}")
    print("faster-whisper에서 사용:")
    print(f'  from faster_whisper import WhisperModel')
    print(f'  model = WhisperModel("{output_dir}", device="cpu")')
    print(f'  segments, info = model.transcribe("audio.wav")')


if __name__ == "__main__":
    main()
