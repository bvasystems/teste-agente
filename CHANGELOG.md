# Changelog

## v0.8.33
- refactor(parsing): update parse function to use GenerationProvider and normalize file extensions
test(parsers): add test for uppercase file extension handling in FileParser

## v0.8.32
- fix(vector_store): prevent pickling errors by avoiding serialization of callables in retrieval_augmented_generation_search

## v0.8.31
- refactor(parsing/docx): fail fast on DOCX→PDF conversion/render errors; remove per-image fallback execution
- perf(parsing/docx): single page screenshot per page when images are present; avoid duplicate OCR in descriptions
- chore(parsing/docx): remove unused imports and resolve lints

## v0.8.30
- feat(parsing): better markdown parsing

## v0.8.29

- feat(providers): implementing provider ID into GenerationProvider subclasses

## v0.8.28
- fix(logging): wrong logging call

## v0.8.27
- refactor: making Langfuse as a parameter

## v0.8.26

- fix(vector_stores): duplicate tools
