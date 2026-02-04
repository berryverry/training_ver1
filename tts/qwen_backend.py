from typing import Any, List, Optional, Union

import torch

try:
    from qwen_tts import Qwen3TTSModel
except ImportError as _e:
    Qwen3TTSModel = None  # type: ignore
    _QWEN_IMPORT_ERROR = _e
else:
    _QWEN_IMPORT_ERROR = None


class Qwen3TTSBackend:
    """
    Description:
        Qwen3-TTS 모델 래퍼. CustomVoice 또는 Base(Voice Clone) 모델 로드
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if Qwen3TTSModel is None:
            raise ImportError(
                "qwen_tts 패키지가 필요합니다.",
                _QWEN_IMPORT_ERROR,
            )
        from .config import MODEL_CUSTOM_VOICE
        name = model_name or MODEL_CUSTOM_VOICE
        device = device_map
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch_dtype
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = Qwen3TTSModel.from_pretrained(
            name,
            device_map=device,
            torch_dtype=dtype,
        )
        self._model_name = name

    def generate_custom_voice(
        self,
        text: List[str],
        language: List[str],
        speaker: List[str],
        instruct: List[str],
    ) -> tuple:
        
        return self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
        )

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple],
        ref_text: str,
        x_vector_only_mode: bool = False,
    ) -> Any:
        """
        Input:
            ref_audio: 참조 음성 파일 경로(또는 URL), 또는 (np.ndarray, sample_rate).
            ref_text: 참조 음성의 대본.
            x_vector_only_mode: True면 ref_text 없이 스피커 임베딩만 사용.
        """
        return self._model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only_mode,
        )

    def generate_voice_clone(
        self,
        text: List[str],
        language: List[str],
        voice_clone_prompt: Any,
    ) -> tuple:
       
        return self._model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )
