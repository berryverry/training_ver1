#!/usr/bin/env bash
# 문장 합성 + 품질 필터.
# 프로젝트 루트에서 실행: cd "프로젝트경로" 후 bash scripts/run_voice_clone.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "$SOX_DIR" ]; then
  for _p in "$SOX_DIR/bin" "$SOX_DIR/src" "$SOX_DIR"; do
    [ -x "$_p/sox" ] && export PATH="$_p:$PATH" && break
  done
fi
for _p in "$PROJECT_ROOT/sox-14.4.1/install/bin" "$PROJECT_ROOT/sox-14.4.1/bin" "$PROJECT_ROOT/sox-14.4.1/src" "$PROJECT_ROOT/sox-14.4.1" \
          "$PROJECT_ROOT/sox-14.4.2/bin" "$PROJECT_ROOT/sox-14.4.2/src" "$PROJECT_ROOT/sox-14.4.2" \
          /opt/homebrew/bin /usr/local/bin; do
  [ -x "$_p/sox" ] && export PATH="$_p:$PATH" && break
done
# sox 실행 파일이 있으면 해당 디렉터리를 PATH에 추가
if ! command -v sox &>/dev/null; then
  for _d in "$PROJECT_ROOT/sox-14.4.1" "$PROJECT_ROOT/sox-14.4.2"; do
    [ ! -d "$_d" ] && continue
    _sox_path="$(find "$_d" -maxdepth 4 -name sox -type f -perm /111 2>/dev/null | head -1)"
    [ -n "$_sox_path" ] && export PATH="$(dirname "$_sox_path"):$PATH" && break
  done
fi

REF_WAV="${REF_WAV:-endAction_clean1.wav}"
# TTS_X_VECTOR_ONLY=1 이면 무시됨(스피커만 사용)
REF_TEXT_FILE="${REF_TEXT_FILE:-scripts/ref_audio.txt}"
TEXT_FILE="${TTS_TEXT_FILE:-scripts/tts_sentences.txt}"
OUT_DIR="${TTS_OUTPUT_DIR:-outputs/tts}"
TTS_X_VECTOR_ONLY="${TTS_X_VECTOR_ONLY:-1}"

if [ ! -f "$REF_WAV" ]; then
  echo "참조 음성 파일이 없습니다: $REF_WAV"
  exit 1
fi

if [ "$TTS_X_VECTOR_ONLY" = "1" ]; then
  REF_TEXT=""
else
  if [ -f "$REF_TEXT_FILE" ]; then
    REF_TEXT="$(tr '\n' ' ' < "$REF_TEXT_FILE" | sed 's/  */ /g; s/^ *//; s/ *$//')"
  else
    REF_TEXT="끝. 촬영 종료."
  fi
  [ -z "$REF_TEXT" ] && REF_TEXT="끝. 촬영 종료."
fi

echo "참조 음성: $REF_WAV"
echo "참조 대본(사용 시): $REF_TEXT"
echo "문장 파일: $TEXT_FILE"
echo "출력 디렉터리: $OUT_DIR"
echo "스피커만 사용(x_vector_only): $TTS_X_VECTOR_ONLY"
echo ""

export TTS_MODE=voice_clone
export TTS_REF_AUDIO="$REF_WAV"
export TTS_REF_TEXT="$REF_TEXT"
export TTS_TEXT_FILE="$TEXT_FILE"
export TTS_OUTPUT_DIR="$OUT_DIR"
export TTS_FILTER_QUALITY=1
export TTS_X_VECTOR_ONLY
# WAV 노이즈 제거 후 합성 (0 이면 끔)
export TTS_DENOISE_REF="${TTS_DENOISE_REF:-1}"
exec python -m tts