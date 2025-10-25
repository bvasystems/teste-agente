from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, override

from agentle.tts.audio_format import AudioFormat
from agentle.tts.output_format_type import OutputFormatType
from agentle.tts.speech_config import SpeechConfig
from agentle.tts.speech_result import SpeechResult
from agentle.tts.tts_provider import TtsProvider
from agentle.tts.voice_settings import VoiceSettings
from agentle.utils.needs import check_modules

if TYPE_CHECKING:
    from elevenlabs import AsyncElevenLabs


class ElevenLabsTtsProvider(TtsProvider):
    _client: AsyncElevenLabs

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        check_modules("elevenlabs")
        from elevenlabs import AsyncElevenLabs

        self._client = AsyncElevenLabs(
            api_key=api_key or os.getenv("ELEVENLABS_API_KEY")
        )

    @override
    async def synthesize_async(self, text: str, config: SpeechConfig) -> SpeechResult:
        from elevenlabs import AsyncElevenLabs
        from elevenlabs.types.voice_settings import (
            VoiceSettings as ElevenLabsVoiceSettings,
        )

        elevenlabs = AsyncElevenLabs()
        audio_stream: AsyncIterator[bytes] = elevenlabs.text_to_speech.convert(
            text=text,
            voice_id=config.voice_id,
            model_id=config.model_id,
            output_format=config.output_format,
            voice_settings=ElevenLabsVoiceSettings(
                stability=config.voice_settings.stability,
                use_speaker_boost=config.voice_settings.use_speaker_boost,
                similarity_boost=config.voice_settings.similarity_boost,
                style=config.voice_settings.style,
                speed=config.voice_settings.speed,
            )
            if config.voice_settings
            else None,
            language_code=config.language_code,
        )

        # Collect all chunks into bytes
        chunks: list[bytes] = []
        async for chunk in audio_stream:
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return SpeechResult(
            audio=audio_base64,
            mime_type=self._get_mime_type(config.output_format),
            format=config.output_format,
        )

    def _get_mime_type(self, output_format: OutputFormatType) -> AudioFormat:
        """Convert ElevenLabs output format to MIME type."""
        if output_format.startswith("mp3_"):
            return "audio/mpeg"
        elif output_format.startswith("pcm_"):
            return "audio/wav"  # or "audio/pcm" depending on your use case
        elif output_format.startswith("ulaw_"):
            return "audio/basic"
        elif output_format.startswith("alaw_"):
            return "audio/basic"
        elif output_format.startswith("opus_"):
            return "audio/opus"
        else:
            return "application/octet-stream"  # fallback


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    tts_provider = ElevenLabsTtsProvider()
    audio = tts_provider.synthesize(
        "Oi, eu sou a Júlia. Assistente pessoal da Dany Braga do estúdio de fotografia. Em que posso ajudar você hoje?",
        config=SpeechConfig(
            voice_id="lWq4KDY8znfkV0DrK8Vb",
            model_id="eleven_v3",
            language_code="pt",
            voice_settings=VoiceSettings(
                stability=0.0,
                use_speaker_boost=None,
                similarity_boost=None,
                style=None,
                speed=None,
            ),
        ),
    )
    with open("audio.mp3", "wb") as file:
        file.write(base64.b64decode(audio.audio))
