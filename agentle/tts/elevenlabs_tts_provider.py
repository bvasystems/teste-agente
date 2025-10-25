import base64
from collections.abc import AsyncIterator
from typing import override

from agentle.tts.audio_format import AudioFormat
from agentle.tts.output_format_type import OutputFormatType
from agentle.tts.speech_config import SpeechConfig
from agentle.tts.speech_result import SpeechResult
from agentle.tts.tts_provider import TtsProvider
from agentle.utils.needs import needs


class ElevenLabsTtsProvider(TtsProvider):
    @override
    @needs("elevenlabs")
    async def synthesize(self, text: str, config: SpeechConfig) -> SpeechResult:
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
