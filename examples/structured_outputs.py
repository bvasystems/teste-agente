"""
Structured Outputs Example

This example demonstrates how to create an agent that returns structured data
using a Pydantic model as a response schema.
"""

from collections.abc import Sequence
from textwrap import dedent
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Discriminator
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.providers.cerebras.cerebras_generation_provider import (
    CerebrasGenerationProvider,
)

load_dotenv()


class FileOutput(BaseModel):
    type: Literal["file"] = Field(default="file")
    title: str
    mime_type: str
    description: str | None = Field(default=None)

    def describe(self, mode: Literal["xml", "markdown"] = "xml") -> str:
        if mode == "markdown":
            return f"""**{self.title}**  \n_MIME type_: `{self.mime_type}`  \n{self.description or ""}"""

        return dedent(f"""
            <FileOutput>
                <Title>{self.title}</Title>
                <MimeType>{self.mime_type}</MimeType>
                <Description>{self.description or ""}</Description>
            </FileOutput>
        """)


class InputOutput(BaseModel):
    """Modelo de entrada para renderização de componentes dinâmicos na UI.

    Atributos
    ---------
    type: Literal["input"]
        Tipo do bloco (fixo como ``"input"``).
    message: str
        Mensagem exibida pela assistente logo acima do título do campo.
    title: str
        Título do campo, visível imediatamente acima do componente de input.
    kind: Literal[
        "text",
        "textarea",
        "number",
        "date",
        "time",
        "datetime",
        "select",
        "autocomplete",
        "checkbox",
        "switch",
        "radio",
        "multiselect",
        "file",
    ]
        Tipo de componente de input que será renderizado na interface.

        • **text** - Campo de texto de uma linha para entradas curtas.
        • **textarea** - Área de texto multilinha para descrições extensas.
        • **number** - Entrada numérica; aceita apenas dígitos e permite step/min/max.
        • **date** - Seletor de **data** (apenas DD/MM/AAAA).
        • **time** - Seletor de **hora** isolada (HH:MM), segundos fixos em ":00".
        • **datetime** - Seletor combinado de **data** e **hora**.
        • **select** - Caixa de seleção (dropdown) com opções pré-definidas.
        • **autocomplete** - Campo com *auto-complete* para listas longas (busca remota).
        • **checkbox** - Caixa de seleção booleana (marcado/desmarcado).
        • **switch** - Variante visual de checkbox, estilo toggle.
        • **radio** - Conjunto de botões radio (uma escolha exclusiva).
        • **multiselect** - Seleção múltipla de opções.
        • **file** - Upload de arquivos (anexos, prints, logs etc.).
    """

    type: Literal["input"] = Field(
        default="input", description="Tipo do bloco (sempre 'input')."
    )

    message: str = Field(
        description="Mensagem mostrada pela assistente antes do título do campo."
    )

    input_title: str = Field(
        description="Título do campo, exibido logo acima do input para o usuário preencher."
    )

    kind: Literal[
        "text",
        "textarea",
        "number",
        "date",
        "time",
        "datetime",
        "select",
        "autocomplete",
        "checkbox",
        "switch",
        "radio",
        "multiselect",
    ] = Field(
        description="""Tipo de input para renderização na UI.

        Valores possíveis:
        - **text**: campo de texto de uma linha.
        - **textarea**: área de texto multilinha.
        - **number**: entrada numérica (somente dígitos).
        - **date**: seletor de data (DD/MM/AAAA).
        - **time**: seletor de hora (HH:MM), segundos fixos em :00.
        - **datetime**: seletor de data e hora juntos.
        - **select**: dropdown de opções fixas.
        - **autocomplete**: campo com busca enquanto digita.
        - **checkbox**: valor booleano (caixa de seleção).
        - **switch**: versão toggle do checkbox.
        - **radio**: grupo de escolhas mutuamente exclusivas.
        - **multiselect**: escolha múltipla de itens.
        """
    )

    default_value: str | float | bool | None = Field(
        default=None,
        description="Um valor padrão para o campo, se conveniente. Caso não "
        + "fornecido, o usuário terá que preencher o input. Se fornecido, "
        + "o usuário pode atualizar o input ou deixar como está. ",
    )

    def describe(self, mode: Literal["xml", "markdown"] = "xml") -> str:
        from textwrap import dedent

        if mode == "markdown":
            return f"""**{self.input_title}**  \n_{self.kind}_  \n{self.message}"""

        return dedent(f"""
            <InputOutput>
                <Message>{self.message}</Message>
                <Title>{self.input_title}</Title>
                <Kind>{self.kind}</Kind>
            </InputOutput>
        """)


class _Option(BaseModel):
    text: str


class Options(BaseModel):
    type: Literal["options"] = Field(default="options")
    options: Sequence[_Option]

    def describe(self, mode: Literal["xml", "markdown"] = "xml") -> str:
        from textwrap import dedent

        if mode == "markdown":
            options_md = "\n".join([f"- {opt.text}" for opt in self.options])
            return f"**Escolha uma opção:**  \n\n{options_md}"

        options_xml = "".join([f"<Option>{opt.text}</Option>" for opt in self.options])
        return dedent(f"""
            <Options>
                {options_xml}
            </Options>
        """)


class _Header(BaseModel):
    type: Literal[
        "os-id",
        "atendimento-id",
        "atendimento-avulso-id",
        "person-id",
        "regular_string",
        "date",
        "number",
    ]

    title: str


class TableOutput(BaseModel):
    type: Literal["table"] = Field(default="table")
    is_preview: bool = Field(default=False)
    headers: Sequence[_Header]
    rows: Sequence[Sequence[str]]

    def describe(self, mode: Literal["xml", "markdown"] = "xml") -> str:
        from textwrap import dedent

        if mode == "markdown":
            headers_md = " | ".join([h.title for h in self.headers])
            separator = " | ".join(["---"] * len(self.headers))
            rows_md = "\n".join([" | ".join(row) for row in self.rows])
            return f"""**Table:**  \n\n| {headers_md} |\n| {separator} |\n{rows_md}"""

        headers_xml = "".join(
            [f"<Header>{h.title}</Title></Header>" for h in self.headers]
        )
        rows_xml = "".join(
            [
                "<Row>" + "".join([f"<Value>{v}</Value>" for v in row]) + "</Row>"
                for row in self.rows
            ]
        )
        return dedent(f"""
        <TableOutput>
            <Headers>{headers_xml}</Headers>
            <Rows>{rows_xml}</Rows>
        </TableOutput>
        """)


class TextOutput(BaseModel):
    type: Literal["text"] = Field(default="text")
    text: str = Field(description="Regular text output.")

    def describe(self, mode: Literal["xml", "markdown"] = "xml") -> str:
        from textwrap import dedent

        if mode == "markdown":
            return self.text
        return dedent(f"""
            <TextOutput>
                <Text>{self.text}</Text>
            </TextOutput>
        """)


OutputType = Annotated[
    FileOutput | InputOutput | TableOutput | TextOutput | Options, Discriminator("type")
]


class Output(BaseModel):
    output: Sequence[OutputType] = Field(description="A resposta final da assistente.")


# Create an agent with the response schema
structured_agent = Agent(
    generation_provider=CerebrasGenerationProvider(),
    model="llama-4-scout-17b-16e-instruct",
    response_schema=Output,
)

# Run the agent with a query that requires structured data
response = structured_agent.run("Give me an example Input of type date.")

print(response.pretty_formatted())
