from typing import Any, Literal, Mapping
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Chunk(BaseModel):
    id: str
    text: str = Field(description="The text of the chunk.")
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    def describe(
        self,
        mode: Literal["xml"] = "xml",
        include_id: bool = False,
        include_metadata: bool = True,
    ) -> str:
        import xml.sax.saxutils as saxutils

        def escape(val: object) -> str:
            return saxutils.escape(str(val))

        lines = ["<Chunk>"]
        if include_id:
            lines.append(f"  <Id>{escape(self.id)}</Id>")
        lines.append(f"  <Text>{escape(self.text)}</Text>")
        if include_metadata and self.metadata:
            lines.append("  <Metadata>")
            for k, v in self.metadata.items():
                lines.append(f"    <{escape(k)}>{escape(v)}</{escape(k)}>")
            lines.append("  </Metadata>")
        lines.append("</Chunk>")
        return "\n".join(lines)
