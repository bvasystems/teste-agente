from rsb.coroutines.run_sync import run_sync
from collections.abc import Sequence
from textwrap import dedent

from html_to_markdown import convert
from playwright.async_api import Geolocation, ViewportSize
from rsb.models import Field
from rsb.models.base_model import BaseModel

from agentle.prompts.models.prompt import Prompt
from agentle.responses.definitions.reasoning import Reasoning
from agentle.responses.responder import Responder
from agentle.utils.needs import needs
from agentle.web.actions.action import Action
from agentle.web.extraction_preferences import ExtractionPreferences
from agentle.web.extraction_result import ExtractionResult

_INSTRUCTIONS = Prompt.from_text(
    dedent("""\
    <character>
    You are a specialist in data extraction and web content analysis. Your role is to act as an intelligent and precise data processor.
    </character>
    
    <request>
    Your task is to analyze the content of a web page provided in Markdown format inside `<markdown>` tags and extract the information requested in the `user_instructions`. You must process the content and return the extracted data in a strictly structured format, according to the requested output schema.
    </request>

    <additions>
    Focus exclusively on the textual content and its structure to identify the data. Ignore irrelevant elements such as script tags, styles, or metadata that do not contain the requested information. If a piece of information requested in `user_instructions` cannot be found in the Markdown content, the corresponding field in the output must be null or empty, as allowed by the schema. Be literal and precise in extraction, avoiding inferences or assumptions not directly supported by the text.
    </additions>
    
    <type>
    The output must be a single valid JSON object that exactly matches the provided data schema. Do not include any text, explanation, comment, or any character outside the JSON object. Your response must start with `{` and end with `}`.
    </type>
    
    <extras>
    Act as an automated extraction tool. Accuracy and schema compliance are your only priorities. Ensure that all required fields in the output schema are filled.
    </extras>
    """)
)

_PROMPT = Prompt.from_text(
    dedent("""\
    {{user_instructions}}

    <markdown>
    {{markdown}}
    </markdown>
    """)
)


# HTML -> MD -> LLM (Structured Output)
class Extractor(BaseModel):
    llm: Responder = Field(..., description="The responder to use for the extractor.")
    reasoning: Reasoning | None = Field(default=None)
    model: str | None = Field(default=None)
    max_output_tokens: int | None = Field(default=None)

    def extract[T: BaseModel](
        self,
        urls: Sequence[str],
        output: type[T],
        prompt: str | None = None,
        extraction_preferences: ExtractionPreferences | None = None,
        ignore_invalid_urls: bool = True,
    ) -> ExtractionResult[T]:
        return run_sync(
            self.extract_async(
                urls, output, prompt, extraction_preferences, ignore_invalid_urls
            )
        )

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
        _actions: Sequence[Action] = _preferences.actions or []

        # Configure proxy if specified
        if _preferences.proxy in ["basic", "stealth"]:
            # You would need to configure actual proxy servers here
            # This is a placeholder for proxy configuration
            pass

        async with async_api.async_playwright() as p:
            browser = await p.chromium.launch(headless=True)

            # Build context options properly based on preferences
            if _preferences.mobile:
                viewport: ViewportSize | None = ViewportSize(width=375, height=667)
                user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
                is_mobile = True
            else:
                viewport = None
                user_agent = None
                is_mobile = None

            # Handle geolocation
            geolocation: Geolocation | None = None
            permissions = None
            if _preferences.location:
                geolocation = Geolocation(
                    latitude=getattr(_preferences.location, "latitude", 0),
                    longitude=getattr(_preferences.location, "longitude", 0),
                )
                permissions = ["geolocation"]

            context = await browser.new_context(
                viewport=viewport,
                user_agent=user_agent,
                is_mobile=is_mobile,
                extra_http_headers=_preferences.headers,
                ignore_https_errors=_preferences.skip_tls_verification,
                geolocation=geolocation,
                permissions=permissions,
            )

            # Block ads if specified
            if _preferences.block_ads:
                await context.route(
                    "**/*",
                    lambda route: route.abort()
                    if route.request.resource_type in ["image", "media", "font"]
                    and any(
                        ad_domain in route.request.url
                        for ad_domain in [
                            "doubleclick.net",
                            "googlesyndication.com",
                            "adservice.google.com",
                            "ads",
                            "analytics",
                            "tracking",
                        ]
                    )
                    else route.continue_(),
                )

            page = await context.new_page()

            for url in urls:
                # Set timeout if specified
                timeout = _preferences.timeout_ms if _preferences.timeout_ms else 30000

                try:
                    await page.goto(url, timeout=timeout)

                    # Wait for specified time if configured
                    if _preferences.wait_for_ms:
                        await page.wait_for_timeout(_preferences.wait_for_ms)

                    # Execute actions
                    for action in _actions:
                        await action.execute(page)

                except Exception as e:
                    if ignore_invalid_urls:
                        print(f"Warning: Failed to load {url}: {e}")
                        continue
                    else:
                        raise

            html = await page.content()

            # Process HTML based on preferences
            if _preferences.remove_base_64_images:
                import re

                html = re.sub(
                    r'<img[^>]+src="data:image/[^"]+"[^>]*>',
                    "",
                    html,
                    flags=re.IGNORECASE,
                )

            # Filter HTML by tags if specified
            if _preferences.include_tags or _preferences.exclude_tags:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "html.parser")

                if _preferences.only_main_content:
                    # Try to find main content area
                    main_content = (
                        soup.find("main")
                        or soup.find("article")
                        or soup.find("div", {"id": "content"})
                        or soup.find("div", {"class": "content"})
                    )
                    if main_content:
                        soup = BeautifulSoup(str(main_content), "html.parser")

                if _preferences.exclude_tags:
                    for tag in _preferences.exclude_tags:
                        for element in soup.find_all(tag):
                            element.decompose()

                if _preferences.include_tags:
                    # Keep only specified tags
                    new_soup = BeautifulSoup("", "html.parser")
                    for tag in _preferences.include_tags:
                        for element in soup.find_all(tag):
                            new_soup.append(element)
                    soup = new_soup

                html = str(soup)

            # Convert to markdown
            markdown = convert(html)

            # Prepare and send prompt
            _prompt = _PROMPT.compile(
                user_instructions=prompt or "Not provided.", markdown=markdown
            )

            response = await self.llm.respond_async(
                input=_prompt,
                model=self.model,
                instructions=_INSTRUCTIONS,
                reasoning=self.reasoning,
                text_format=output,
            )

            await browser.close()

            return ExtractionResult[T](
                urls=urls,
                html=html,
                markdown=markdown,
                extraction_preferences=_preferences,
                output_parsed=response.output_parsed,
            )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    site_uniube = "https://uniube.br/"

    class PossiveisRedirecionamentos(BaseModel):
        possiveis_redirecionamentos: list[str]

    extractor = Extractor(
        llm=Responder.openai(),
        model="gpt-5-nano",
    )

    # Example with custom extraction preferences
    preferences = ExtractionPreferences(
        only_main_content=True,
        wait_for_ms=2000,
        block_ads=True,
        remove_base_64_images=True,
        timeout_ms=15000,
    )

    result = extractor.extract(
        urls=[site_uniube],
        output=PossiveisRedirecionamentos,
        prompt="Extract the possible redirects from the page.",
        extraction_preferences=preferences,
    )

    for link in result.output_parsed.possiveis_redirecionamentos:
        print(f"Link: {link}")
