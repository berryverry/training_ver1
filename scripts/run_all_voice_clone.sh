#!/usr/bin/env bash
# 모든 WAV를 써서 tts_sentences.txt 각 줄을 합성하고,
# outputs/tts/<참조wav이름>/ 폴더에 저장.
# 프로젝트 루트에서 실행: cd "프로젝트경로" 후 bash scripts/run_all_voice_clone.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

WAV_DIR="${WAV_DIR:-$PROJECT_ROOT}"
# 문장 파일 (scripts/tts_sentences.txt)
TTS_TEXT_FILE="${TTS_TEXT_FILE:-scripts/tts_sentences.txt}"

if [ ! -f "$TTS_TEXT_FILE" ]; then
  echo "문장 파일이 없습니다: $TTS_TEXT_FILE"
  exit 1
fi

echo "WAV 디렉터리: $WAV_DIR"
echo "문장 파일: $TTS_TEXT_FILE"
echo "출력 루트: outputs/tts/<참조이름>/"
echo ""

count=0
for wav in "$WAV_DIR"/*.wav; do
  [ -f "$wav" ] || continue
  base="$(basename "$wav" .wav)"
  out_dir="outputs/tts/$base"
  count=$((count + 1))
  echo "---- [$count] 참조: $wav → $out_dir"
  REF_WAV="$wav" TTS_OUTPUT_DIR="$out_dir" TTS_TEXT_FILE="$TTS_TEXT_FILE" bash "$SCRIPT_DIR/run_voice_clone.sh"
  echo ""
done

if [ "$count" -eq 0 ]; then
  echo "WAV 파일이 없습니다: $WAV_DIR/*.wav"
  exit 1
fi

echo "완료: 참조 WAV $count개에 대해 합성됨. outputs/tts/<이름>/ 에 저장됨."