from collections.abc import Sequence
from typing import Literal

from html_to_markdown import convert
from rsb.models import Field
from rsb.models.base_model import BaseModel

from agentle.prompts.models.prompt import Prompt
from agentle.responses.definitions.reasoning import Reasoning
from agentle.responses.responder import Responder
from agentle.utils.needs import needs
from agentle.web.extraction_preferences import ExtractionPreferences
from agentle.web.extraction_result import ExtractionResult


# HTML -> MD -> LLM (Structured Output)
class Extractor(BaseModel):
    responder: Responder = Field(
        ..., description="The responder to use for the extractor."
    )
    reasoning: Reasoning | None = Field(default=None)
    model: str | None = Field(default=None)

    @needs("playwright")
    async def extract_async[T: BaseModel](
        self,
        urls: Sequence[str],
        output: type[T],
        prompt: str | None = None,
        extraction_preferences: ExtractionPreferences | None = None,
        ignore_invalid_urls: bool = True,
    ) -> ExtractionResult[T]:
        from playwright import async_api

        _preferences = extraction_preferences or ExtractionPreferences()
        _actions = _preferences.actions or []

        async with async_api.async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            for url in urls:
                await page.goto(url, timeout=10000)

                for action in _actions:
                    await action.execute(page)

            html = await page.content()
            markdown = convert(html)

            _prompt = Prompt.from_text(markdown)

            response = await self.responder.respond_async(
                input=_prompt,
                model=self.model,
                instructions=prompt,
                reasoning=self.reasoning,
                text_format=output,
            )

            return ExtractionResult[T](
                urls=urls,
                html=html,
                markdown=markdown,
                extraction_preferences=_preferences,
                result=response.output_parsed,
            )
