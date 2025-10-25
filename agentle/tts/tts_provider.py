import abc

from agentle.tts.speech_config import SpeechConfig
from agentle.tts.speech_result import SpeechResult


class TtsProvider(abc.ABC):
    @abc.abstractmethod
    async def synthesize(self, text: str, config: SpeechConfig) -> SpeechResult: ...
