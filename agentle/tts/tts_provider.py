import abc

from agentle.tts.real_time.definitions.speech_config import SpeechConfig
from agentle.tts.real_time.definitions.speech_result import SpeechResult


class TtsProvider(abc.ABC):
    async def synthesize(
        self, text: str, config: SpeechConfig | None = None
    ) -> SpeechResult: ...
