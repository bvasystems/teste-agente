from typing import NotRequired, TypedDict


class _RecursiveCharacterTextSplitterConfig(TypedDict):
    chunk_size: NotRequired[int]
    chunk_overlap: NotRequired[int]


type ChunkingConfig = _RecursiveCharacterTextSplitterConfig
