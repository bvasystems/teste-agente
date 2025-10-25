from rsb.models import BaseModel, Field

from agentle.tts.audio_format import AudioFormat
from agentle.tts.output_format_type import OutputFormatType


class SpeechResult(BaseModel):
    audio: str = Field(...)
    """the raw audio bytes in base-64 format"""

    mime_type: AudioFormat = Field(...)
    """`audio/mpeg`, `audio/wav`, `audio/opus`"""

    format: OutputFormatType = Field(...)
    """The original format string like "mp3_44100_128"""
