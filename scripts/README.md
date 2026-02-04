# Whisper 파인튜닝 실행 방법

## 0. 학습 데이터 생성 (Voice clone TTS → WAV + train.csv)

Whisper 파인튜닝용 로컬 데이터는 **Voice clone TTS**로 문장을 읽은 WAV와 그에 맞는 `train.csv`로 구성됩니다.

### 1. Voice clone으로 WAV + train.csv 한 번에 생성

문장 텍스트 파일과 참조 음성(및 참조 문장)을 주면 TTS가 WAV를 생성하고, 같은 폴더에 `train.csv`를 씁니다.

```bash
cd /Users/user/Downloads/training1
source .venv/bin/activate

# 문장 파일: 한 줄에 한 문장 (순서대로 WAV와 매칭됨)
# 예: scripts/tts_sentences.txt

TTS_MODE=voice_clone \
TTS_TEXT_FILE=scripts/tts_sentences.txt \
TTS_OUTPUT_DIR=outputs/tts/내데이터이름 \
TTS_REF_AUDIO=path/to/reference.wav \
TTS_REF_TEXT="참조 음성에서 읽은 문장" \
TTS_LANGUAGE=Korean \
python -m tts
```

- **TTS_REF_AUDIO**: 목소리 복제용 참조 WAV 경로 (필수)
- **TTS_REF_TEXT**: 참조 WAV에 대응하는 텍스트. 없으면 `TTS_X_VECTOR_ONLY=1` 로 설정
- **TTS_OUTPUT_DIR**: 생성 WAV와 `train.csv`가 저장될 폴더 (예: `outputs/tts/endAction_clean1`)
- **TTS_WRITE_MANIFEST**: 기본값 `1` → `train.csv` 생성. `0`이면 CSV 미생성
- **TTS_FILTER_QUALITY=1**: 품질 검사 후 통과한 것만 `train.csv`에 넣고, 탈락은 `rejected.csv`에 기록
- **참조 음성 3초 초과**: 참조 WAV가 3초를 넘어도 오류 없이 앞 3초만 사용. `TTS_REF_MAX_DURATION_SEC=5` 등으로 변경 가능.
- **긴 문장 합성(3초 이상)**: `TTS_MAX_NEW_TOKENS=8192`(기본)으로 긴 문장도 끝까지 생성. 더 길게는 `TTS_MAX_NEW_TOKENS=16384` 등으로 설정.
- **말빠르기·억양·어조 다양화**: 배치마다 `temperature`, `top_p`, `repetition_penalty`, `subtalker_temperature`/`subtalker_top_p`/`subtalker_top_k` 랜덤 적용. 끄려면 `TTS_VARY_PROSODY=0`.  
  범위 조정: `TTS_TEMPERATURE_MIN`/`MAX`, `TTS_REPETITION_PENALTY_MIN`/`MAX`, `TTS_SUBTALKER_*` 환경변수. `TTS_BATCH_SIZE=4` 로 줄이면 문장마다 어조 차이 더 커짐.
- **여러 참조(남/녀 등)**: 문장마다 다른 목소리 사용. **반드시 실제 남성/여성 음성 파일**을 각각 넣어야 남녀가 골고루 나옴.
  - **남/녀 50:50**: `REF_WAV_MALE=남성.wav REF_WAV_FEMALE=여성.wav` 로 지정. 기본으로 문장의 절반은 남성·절반은 여성 참조 사용 (`TTS_BALANCE_REFS=0` 이면 완전 랜덤).
  - 또는 `REF_WAV="남성.wav,여성.wav"` 처럼 쉼표로 지정.

#### 실행 예시 (어조·성별·속도 설정 포함)

**참조 1개 + 어조 다양화만**
```bash
cd /Users/user/gitver3/training_ver1   # 프로젝트 루트로
source .venv/bin/activate
REF_WAV=참조.wav \
TTS_OUTPUT_DIR=outputs/tts/내이름 \
TTS_VARY_PROSODY=1 \
TTS_BATCH_SIZE=16 \
bash scripts/run_voice_clone.sh
```

**어조·억양 다양화 (권장)**
```bash
cd /Users/user/gitver3/training_ver1
source .venv/bin/activate
REF_WAV_MALE=남성참조.wav \
REF_WAV_FEMALE=여성참조.wav \
TTS_OUTPUT_DIR=outputs/tts/랜덤 \
TTS_VARY_PROSODY=1 \
TTS_BALANCE_REFS=1 \
TTS_BATCH_SIZE=16 \
bash scripts/run_voice_clone.sh
```

**어조 변화 극대화 (배치 작게)**
```bash
REF_WAV_MALE=남성.wav REF_WAV_FEMALE=여성.wav \
TTS_OUTPUT_DIR=outputs/tts/다양 \
TTS_VARY_PROSODY=1 \
TTS_BATCH_SIZE=4 \
bash scripts/run_voice_clone.sh
```

**속도 우선 (어조 다양화 유지)**
```bash
REF_WAV=참조.wav \
TTS_OUTPUT_DIR=outputs/tts/빠름 \
TTS_BATCH_SIZE=24 \
bash scripts/run_voice_clone.sh
```

생성 결과: `outputs/tts/내데이터이름/` 아래에 `voice_clone_0001.wav`, `voice_clone_0002.wav`, ... 와 `train.csv` (컬럼: path, sentence).

**속도 올리기**: 기본 배치 크기는 16. 더 빠르게 하려면 `TTS_BATCH_SIZE=24` 또는 `32`. (Apple Silicon은 MPS 자동 사용. NVIDIA GPU는 `pip install flash-attn` 권장.)

### 2. 이미 만든 WAV 폴더에서 train.csv만 만들기

WAV는 이미 있고, 문장 파일만 있으면 품질 검사 후 `train.csv` / `rejected.csv`만 생성할 수 있습니다.  
WAV 파일명은 `voice_clone_0001.wav` 형태(접두어 + 숫자 + .wav)이고, **텍스트 파일의 줄 순서와 1:1**이어야 합니다.

```bash
TTS_MODE=filter_quality \
TTS_OUTPUT_DIR=outputs/tts/기존WAV폴더 \
TTS_TEXT_FILE=scripts/tts_sentences.txt \
python -m tts
```

---

## 3. 로컬 데이터로 실행 (Voice clone으로 만든 WAV + train.csv)

Voice clone으로 이미 `outputs/tts/<이름>/train.csv` 가 있다면, 그 폴더를 데이터로 쓸 수 있습니다.

```bash
cd /Users/user/Downloads/training1

# 가상환경 사용
source .venv/bin/activate   # 또는: PATH="$(pwd)/.venv/bin:$PATH"

# 한 폴더(예: endAction_clean1)만 훈련
python finetune_whisper.py \
  --data_dir outputs/tts/endAction_clean1 \
  --train_csv train.csv \
  --output_dir ./whisper-finetuned \
  --model_name openai/whisper-small \
  --language Korean \
  --num_epochs 3 \
  --batch_size 8
```

- **--data_dir**: `train.csv`와 WAV 파일들이 있는 폴더 (예: `outputs/tts/endAction_clean1`)
- **--train_csv**: 훈련용 CSV 파일명 (기본 `train.csv`). CSV 컬럼: **path**(또는 audio), **sentence**(또는 text)
- **--output_dir**: 학습된 모델 저장 경로 (기본 `./whisper-finetuned`)
- **--model_name**: 사전학습 모델 (예: `openai/whisper-base`, `openai/whisper-small`)
- **--language**: Korean / English 등
- **--num_epochs**, **--batch_size** 등은 필요에 따라 조정

---

## 4. Hugging Face 데이터셋으로 실행

로컬 CSV 없이 HF 데이터셋만 쓸 때:

```bash
cd /Users/user/Downloads/training1
source .venv/bin/activate

python finetune_whisper.py \
  --dataset "Korea-Audio-Tasks/zeroth_korean" \
  --dataset_config clean \
  --output_dir ./whisper-finetuned \
  --model_name openai/whisper-small \
  --language Korean \
  --num_epochs 2
```

---

## 5. 학습 후 CTranslate2로 변환 (feat. faster-whisper)

```bash
python convert_to_ct2.py ./whisper-finetuned --output_dir ./whisper-ct2
```

이후 faster-whisper에서는 `./whisper-ct2` 경로를 지정해 사용하면 됩니다.