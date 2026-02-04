import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from dataclasses import dataclass
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
import librosa


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------

def load_local_dataset(data_dir: str, train_csv: str, eval_csv: Optional[str]) -> DatasetDict:
    """
    Description:
        로컬 폴더의 CSV로 데이터셋 로드. CSV 컬럼: audio(또는 path) 파일 경로, sentence(또는 text) 정답 텍스트.
        eval_csv가 없으면 train에서 10% 분리해 검증용으로 사용.

    Input:
        data_dir: 데이터 폴더 경로
        train_csv: 훈련용 CSV 파일명 (data_dir 기준)
        eval_csv: 검증용 CSV 파일명 (없으면 train에서 분리)

    Returns:
        DatasetDict: {"train": Dataset, "eval": Dataset}
    """
    data_path = Path(data_dir)
    train_file = data_path / train_csv
    if not train_file.exists():
        raise FileNotFoundError(f"훈련 CSV를 찾을 수 없습니다: {train_file}")

    train_ds = load_dataset("csv", data_files=str(train_file), split="train")

    if eval_csv and (data_path / eval_csv).exists():
        eval_ds = load_dataset("csv", data_files=str(data_path / eval_csv), split="train")
    else:
        # 검증용으로 train 일부 분리 (10%)
        split = train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]

    return DatasetDict({"train": train_ds, "eval": eval_ds})


def prepare_dataset_factory(processor: WhisperProcessor, lang: Optional[str], task: str = "transcribe"):
    """
    Description:
        prepare_dataset 함수를 반환. processor, lang, task를 클로저로 캡처하여
        배치마다 input_features와 labels를 생성한다.

    Input:
        processor: WhisperProcessor (feature_extractor, tokenizer 사용)
        lang: 타깃 언어 (Whisper tokenizer용)
        task: "transcribe" 또는 "translate"

    Returns:
        Callable: prepare_dataset(batch) -> Dict with input_features, labels
    """

    def prepare_dataset(batch: Dict[str, Any]) -> Dict[str, Any]:
        # 오디오: datasets Audio 컬럼은 {"array": ..., "sampling_rate": ...} 또는 path에서 로드됨
        audio = batch.get("audio")
        if audio is not None:
            if isinstance(audio, dict):
                array = audio.get("array")
                sr = audio.get("sampling_rate", 16000)
            else:
                array = audio
                sr = 16000
        else:
            array = batch.get("input_values")
            sr = 16000
        if array is None:
            raise ValueError("배치에 'audio' 컬럼이 필요합니다 (Audio(sampling_rate=16000) 캐스팅 후).")

        if sr != 16000:
            import librosa
            array = librosa.resample(array.astype(float), orig_sr=sr, target_sr=16000)

        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=16000, return_tensors="pt"
        ).input_features[0].squeeze(0).numpy()

        text = batch.get("sentence", batch.get("text", ""))
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        batch["labels"] = processor.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        return batch

    return prepare_dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Description:
        Seq2Seq 훈련용 데이터 콜레이터. input_features와 labels를 패딩하고,
        labels의 패딩 위치는 -100으로 마스크한다.

    Input:
        processor: WhisperProcessor
        decoder_start_token_id: 디코더 시작 토큰 ID

    Returns:
        None (dataclass)
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], Any]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Description:
            배치 내 샘플들의 input_features와 labels를 패딩해 하나의 텐서 배치로 만든다.

        Input:
            features: List of dict with "input_features", "labels"

        Returns:
            Dict[str, torch.Tensor]: "input_features", "labels" 등
        """
        # input_features: 고정 30초 스펙트로그램이면 그냥 스택
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # labels: 패딩 후 -100으로 마스크
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt", padding=True
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # decoder start token이 맨 앞에 있으면 제거 (Trainer가 자동 붙임)
        if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Description:
        명령줄 인자를 파싱하고, 로컬/HF 데이터셋 로드 후 Whisper를 파인튜닝한다.
        모델과 processor를 output_dir에 저장한다.

    Input:
        None (argparse로 --model_name, --data_dir, --output_dir 등 처리)

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Whisper 파인튜닝 (Hugging Face Transformers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-small",
        help="사전 학습 모델 이름 또는 경로 (예: openai/whisper-small, openai/whisper-base)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="로컬 데이터 폴더. train.csv(, eval.csv) 포함. CSV 컬럼: audio 또는 path, sentence",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="train.csv",
        help="훈련용 CSV 파일명 (data_dir 기준)",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=None,
        help="검증용 CSV 파일명 (없으면 train에서 10%% 분리)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Hugging Face 데이터셋 이름 (예: mozilla-foundation/common_voice_11_0)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="HF 데이터셋 config (예: ko, hi)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./whisper-finetuned",
        help="모델 저장 경로",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Korean",
        help="타깃 언어 (Whisper tokenizer용, 예: Korean, English)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="transcribe 또는 translate",
    )
    parser.add_argument(
        "--num_epochs",
        type=float,
        default=3.0,
        help="훈련 에폭 수",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="디바이스당 배치 크기",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="학습률",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup 스텝 수",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="체크포인트 저장 주기(스텝)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="평가 주기(스텝)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="최대 스텝 (-1이면 num_epochs 기준)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="FP16 사용 (기본 True)",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=False,
        help="그래디언트 체크포인팅 (메모리 절약). Mac MPS에서는 끄는 것이 안정적.",
    )
    parser.add_argument(
        "--report_tensorboard",
        action="store_true",
        default=False,
        help="TensorBoard 로깅 사용 (기본 끔, 디렉터리 오류 방지)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="훈련 후 Hugging Face Hub에 업로드",
    )
    args = parser.parse_args()

    # 데이터셋 로드
    if args.data_dir:
        print(f"로컬 데이터 로드: {args.data_dir}")
        dataset = load_local_dataset(args.data_dir, args.train_csv, args.eval_csv)
        data_dir_abs = os.path.abspath(args.data_dir)
        if "path" not in dataset["train"].column_names and "audio" not in dataset["train"].column_names:
            raise ValueError("CSV에 'path' 또는 'audio'(파일 경로) 컬럼과 'sentence'(또는 'text') 컬럼이 필요합니다.")
        audio_col = "path" if "path" in dataset["train"].column_names else "audio"

        def resolve_paths(rows):
            out = {audio_col: []}
            for p in rows[audio_col]:
                out[audio_col].append(os.path.join(data_dir_abs, p) if p and not os.path.isabs(p) else p)
            return out

        dataset = dataset.map(resolve_paths, batched=True, num_proc=1, desc="경로 해석")

        # 문자열 path → 오디오 배열 로드 (librosa: wav/mp3/m4a 등 지원, m4a는 ffmpeg 필요)
        def load_audio_from_path(rows):
            arrays, srs = [], []
            for path in rows[audio_col]:
                arr, sr = librosa.load(path, sr=None, mono=True, dtype="float32")
                if sr != 16000:
                    arr = librosa.resample(arr.astype(float), orig_sr=sr, target_sr=16000)
                    sr = 16000
                arrays.append(arr)
                srs.append(sr)
            return {"audio": [{"array": a, "sampling_rate": s} for a, s in zip(arrays, srs)]}

        dataset = dataset.map(
            load_audio_from_path,
            batched=True,
            num_proc=1,
            desc="오디오 로드",
            remove_columns=[audio_col],
        )
        if audio_col == "path":
            pass  # 이미 audio 컬럼으로 채움
        if "text" in dataset["train"].column_names and "sentence" not in dataset["train"].column_names:
            dataset = dataset.rename_column("text", "sentence")
    elif args.dataset:
        print(f"HF 데이터셋 로드: {args.dataset} (config={args.dataset_config})")
        dataset = load_dataset(args.dataset, args.dataset_config or "ko", trust_remote_code=True)
        if "validation" in dataset:
            dataset = DatasetDict({"train": dataset["train"], "eval": dataset["validation"]})
        elif "test" in dataset:
            dataset = DatasetDict({"train": dataset["train"], "eval": dataset["test"]})
        else:
            split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            dataset = DatasetDict({"train": split["train"], "eval": split["test"]})
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        if "sentence" not in dataset["train"].column_names and "text" in dataset["train"].column_names:
            dataset = dataset.rename_column("text", "sentence")
    else:
        print("오류: --data_dir 또는 --dataset 중 하나를 지정하세요.", file=sys.stderr)
        sys.exit(1)

    # Processor & Model
    print(f"Processor/모델 로드: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    lang_code = args.dataset_config or {"Korean": "ko", "English": "en"}.get(args.language, (args.language[:2].lower() if len(args.language) >= 2 else "en"))
    model.generation_config.language = lang_code
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None

    # 데이터 전처리: input_features, labels
    prepare_fn = prepare_dataset_factory(processor, args.language, args.task)
    dataset = dataset.map(
        prepare_fn,
        remove_columns=dataset["train"].column_names,
        num_proc=2,
        desc="전처리",
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # WER 메트릭
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        """
        Description:
            예측과 레이블로 WER(Word Error Rate)을 계산해 {"wer": float} 반환.

        Input:
            pred: EvalPrediction (predictions, label_ids)

        Returns:
            Dict[str, float]: {"wer": wer_percent}
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["tensorboard"] if args.report_tensorboard else [],
        push_to_hub=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("훈련 시작.")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"모델 저장 완료: {args.output_dir}")
    print("faster-whisper에서 사용하려면 CTranslate2로 변환 후 해당 경로를 WhisperModel에 지정하세요.")


if __name__ == "__main__":
    main()
