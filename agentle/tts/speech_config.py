from typing import Optional

from rsb.models import BaseModel, Field

from agentle.tts.output_format_type import OutputFormatType
from agentle.tts.voice_settings import VoiceSettings


class SpeechConfig(BaseModel):
    voice_id: str
    model_id: Optional[str] = Field(default=None)
    output_format: OutputFormatType = Field(default="mp3_22050_32")
    language_code: Optional[str] = Field(default=None)
    voice_settings: Optional[VoiceSettings] = Field(default=None)
