from typing import Optional

from rsb.models import BaseModel, Field


class VoiceSettings(BaseModel):
    stability: Optional[float] = Field(default=None)
    """
    Determines how stable the voice is and the randomness between each generation. Lower values introduce broader emotional range for the voice. Higher values can result in a monotonous voice with limited emotion.
    """

    use_speaker_boost: Optional[bool] = Field(default=None)
    """
    This setting boosts the similarity to the original speaker. Using this setting requires a slightly higher computational load, which in turn increases latency.
    """

    similarity_boost: Optional[float] = Field(default=None)
    """
    Determines how closely the AI should adhere to the original voice when attempting to replicate it.
    """

    style: Optional[float] = Field(default=None)
    """
    Determines the style exaggeration of the voice. This setting attempts to amplify the style of the original speaker. It does consume additional computational resources and might increase latency if set to anything other than 0.
    """

    speed: Optional[float] = Field(default=None)
    """
    Adjusts the speed of the voice. A value of 1.0 is the default speed, while values less than 1.0 slow down the speech, and values greater than 1.0 speed it up.
    """
