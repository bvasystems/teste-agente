from __future__ import annotations

import asyncio
from collections.abc import Sequence
from textwrap import dedent
from typing import TYPE_CHECKING

from html_to_markdown import convert
from rsb.coroutines.run_sync import run_sync
from rsb.models import Field
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict

from agentle.generations.models.generation.generation import Generation
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.prompts.models.prompt import Prompt
from agentle.responses.definitions.reasoning import Reasoning
from agentle.responses.responder import Responder
from agentle.utils.needs import needs
from agentle.web.actions.action import Action
from agentle.web.extraction_preferences import ExtractionPreferences
from agentle.web.extraction_result import ExtractionResult

if TYPE_CHECKING:
    from playwright.async_api import Browser, Geolocation, ViewportSize


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
    llm: Responder | GenerationProvider = Field(
        ..., description="The responder to use for the extractor."
    )
    reasoning: Reasoning | None = Field(default=None)
    model: str | None = Field(default=None)
    max_output_tokens: int | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def extract_markdown(
        self,
        browser: Browser,
        urls: Sequence[str],
        extraction_preferences: ExtractionPreferences | None = None,
        ignore_invalid_urls: bool = True,
    ) -> tuple[str, str]:
        return run_sync(
            self.extract_markdown_async,
            browser=browser,
            urls=urls,
            extraction_preferences=extraction_preferences,
            ignore_invalid_urls=ignore_invalid_urls,
        )

    async def extract_markdown_async(
        self,
        browser: Browser,
        urls: Sequence[str],
        extraction_preferences: ExtractionPreferences | None = None,
        ignore_invalid_urls: bool = True,
    ) -> tuple[str, str]:
        _preferences = extraction_preferences or ExtractionPreferences()
        _actions: Sequence[Action] = _preferences.actions or []

        # Configure proxy if specified
        if _preferences.proxy in ["basic", "stealth"]:
            # You would need to configure actual proxy servers here
            # This is a placeholder for proxy configuration
            pass

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

        # Process HTML based on preferences - consolidate all BeautifulSoup operations
        if (
            _preferences.remove_base_64_images
            or _preferences.include_tags
            or _preferences.exclude_tags
            or _preferences.only_main_content
        ):
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove base64 images first
            if _preferences.remove_base_64_images:
                import re

                # Debug: Check what we have before processing
                all_imgs = soup.find_all("img")
                print(f"DEBUG: Found {len(all_imgs)} img tags total")
                base64_count = 0
                for img in all_imgs:
                    src = img.attrs.get("src") if hasattr(img, "attrs") else None  # type: ignore[union-attr]
                    if isinstance(src, str) and "data:image/" in src:
                        base64_count += 1
                        print(f"DEBUG: Found base64 img: {src[:100]}...")
                print(f"DEBUG: {base64_count} images have base64 data")

                # First, remove any anchor tags that contain img children with base64
                # (must be done before removing img tags themselves)
                removed_anchors = 0
                for a_tag in soup.find_all("a"):
                    imgs = a_tag.find_all("img")  # type: ignore[union-attr]
                    for img in imgs:
                        src = img.attrs.get("src") if hasattr(img, "attrs") else None  # type: ignore[union-attr]
                        if isinstance(src, str) and src.startswith("data:image/"):
                            # Remove the entire anchor tag if it contains base64 image
                            a_tag.decompose()
                            removed_anchors += 1
                            break
                print(
                    f"DEBUG: Removed {removed_anchors} anchor tags with base64 images"
                )

                # Remove standalone img tags with base64 src
                removed_imgs = 0
                for img in soup.find_all("img"):
                    src = img.attrs.get("src") if hasattr(img, "attrs") else None  # type: ignore[union-attr]
                    if isinstance(src, str) and src.startswith("data:image/"):
                        img.decompose()
                        removed_imgs += 1
                print(f"DEBUG: Removed {removed_imgs} standalone img tags")

                # Remove any element with base64 in href (like anchor tags with image data)
                for elem in soup.find_all(attrs={"href": True}):
                    href = elem.attrs.get("href") if hasattr(elem, "attrs") else None  # type: ignore[union-attr]
                    if isinstance(href, str) and href.startswith("data:image/"):
                        elem.decompose()

                # Remove any element with base64 in style attribute
                for elem in soup.find_all(attrs={"style": True}):
                    style = elem.attrs.get("style") if hasattr(elem, "attrs") else None  # type: ignore[union-attr]
                    if isinstance(style, str) and "data:image/" in style:
                        elem.decompose()

                # Remove SVG tags (they often contain base64 or are converted to base64 by markdown)
                for svg in soup.find_all("svg"):
                    svg.decompose()

                # Remove any anchor tags that contain SVG children
                for a_tag in soup.find_all("a"):
                    if a_tag.find("svg"):  # type: ignore[union-attr]
                        a_tag.decompose()

                # Final check: see if any base64 remains in the HTML string
                html_str = str(soup)
                remaining = len(re.findall(r'data:image/[^"\')\s]+', html_str))
                print(
                    f"DEBUG: After processing, {remaining} base64 data URIs remain in HTML"
                )

            # Extract main content if requested
            if _preferences.only_main_content:
                main_content = (
                    soup.find("main")
                    or soup.find("article")
                    or soup.find("div", {"id": "content"})
                    or soup.find("div", {"class": "content"})
                )
                if main_content:
                    soup = main_content  # type: ignore[assignment]

            # Exclude specific tags
            if _preferences.exclude_tags:
                for tag in _preferences.exclude_tags:
                    for element in soup.find_all(tag):  # type: ignore[union-attr]
                        element.decompose()

            # Include only specific tags
            if _preferences.include_tags:
                new_soup = BeautifulSoup("", "html.parser")
                for tag in _preferences.include_tags:
                    for element in soup.find_all(tag):  # type: ignore[union-attr]
                        new_soup.append(element)  # type: ignore[arg-type]
                soup = new_soup

            html = str(soup)

        # Convert to markdown
        markdown = convert(html)
        return html, markdown

    def extract[T: BaseModel](
        self,
        browser: Browser,
        urls: Sequence[str],
        output: type[T],
        prompt: str | None = None,
        extraction_preferences: ExtractionPreferences | None = None,
        ignore_invalid_urls: bool = True,
    ) -> ExtractionResult[T]:
        return run_sync(
            self.extract_async(
                browser=browser,
                urls=urls,
                output=output,
                prompt=prompt,
                extraction_preferences=extraction_preferences,
                ignore_invalid_urls=ignore_invalid_urls,
            )
        )

    @needs("playwright")
    async def extract_async[T: BaseModel](
        self,
        browser: Browser,
        urls: Sequence[str],
        output: type[T],
        prompt: str | None = None,
        extraction_preferences: ExtractionPreferences | None = None,
        ignore_invalid_urls: bool = True,
    ) -> ExtractionResult[T]:
        _preferences = extraction_preferences or ExtractionPreferences()

        html, markdown = await self.extract_markdown_async(
            browser=browser,
            urls=urls,
            extraction_preferences=_preferences,
            ignore_invalid_urls=ignore_invalid_urls,
        )

        # Prepare and send prompt
        _prompt = _PROMPT.compile(
            user_instructions=prompt or "Not provided.", markdown=markdown
        )

        if isinstance(self.llm, GenerationProvider):
            response = await self.llm.generate_by_prompt_async(
                prompt=_prompt,
                model=self.model,
                developer_prompt=_INSTRUCTIONS,
                response_schema=output,
            )
        else:
            response = await self.llm.respond_async(
                input=_prompt,
                model=self.model,
                instructions=_INSTRUCTIONS,
                reasoning=self.reasoning,
                text_format=output,
            )

        output_parsed = (
            response.parsed
            if isinstance(response, Generation)
            else response.output_parsed
        )

        await browser.close()

        return ExtractionResult[T](
            urls=urls,
            html=html,
            markdown=markdown,
            extraction_preferences=_preferences,
            output_parsed=output_parsed,
        )


async def test() -> None:
    from dotenv import load_dotenv
    from playwright import async_api

    load_dotenv()

    site_uniube = "https://uniube.br/"

    class PossiveisRedirecionamentos(BaseModel):
        possiveis_redirecionamentos: list[str]

    extractor = Extractor(
        llm=Responder.openrouter(),
        model="google/gemini-2.5-flash",
    )

    # Example with custom extraction preferences
    preferences = ExtractionPreferences(
        only_main_content=True,
        wait_for_ms=2000,
        block_ads=True,
        remove_base_64_images=True,
        timeout_ms=15000,
    )

    async with async_api.async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        result = await extractor.extract_async(
            browser=browser,
            urls=[site_uniube],
            output=PossiveisRedirecionamentos,
            prompt="Extract the possible redirects from the page.",
            extraction_preferences=preferences,
        )

        for link in result.output_parsed.possiveis_redirecionamentos:
            print(f"Link: {link}")


if __name__ == "__main__":
    asyncio.run(test())
