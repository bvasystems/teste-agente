from typing import Annotated

from pydantic import Field

from agentle.web.actions.click import Click
from agentle.web.actions.execute_javascript import ExecuteJavascript
from agentle.web.actions.generate_pdf import GeneratePdf
from agentle.web.actions.press_a_key import PressAKey
from agentle.web.actions.scrape import Scrape
from agentle.web.actions.screenshot import Screenshot
from agentle.web.actions.scroll import Scroll
from agentle.web.actions.wait import Wait
from agentle.web.actions.write_text import WriteText

type Action = Annotated[
    Wait
    | Screenshot
    | Click
    | WriteText
    | PressAKey
    | Scroll
    | Scrape
    | ExecuteJavascript
    | GeneratePdf,
    Field(discriminator="type"),
]
