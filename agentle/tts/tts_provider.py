import abc

from rsb.coroutines.run_sync import run_sync

from agentle.tts.speech_config import SpeechConfig
from agentle.tts.speech_result import SpeechResult


class TtsProvider(abc.ABC):
    def synthesize(self, text: str, config: SpeechConfig) -> SpeechResult:
        return run_sync(self.synthesize_async, text=text, config=config)

    @abc.abstractmethod
    async def synthesize_async(
        self, text: str, config: SpeechConfig
    ) -> SpeechResult: ...
