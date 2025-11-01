# Changelog

## v0.9.27
feat(whatsapp): Add structured response base model for WhatsApp bot

- Create WhatsAppResponseBase model to standardize bot response structure
- Enhance WhatsAppBot to support generic structured output schemas
- Update _send_response method to handle structured response parsing
- Add type hints and documentation for new response handling mechanism
- Improve logging for structured response processing
- Enable more flexible and type-safe bot response generation
This change introduces a base model for WhatsApp bot responses that ensures a consistent response structure while allowing for extensible, type-safe output schemas.


## v0.9.26
feat(api): Enhance API endpoint and function name generation

- Improve aiohttp session handling in `make_single_request()` to prevent connection errors
- Add robust function name generation for OpenAPI spec endpoints
- Create new test script to validate function name generation for OpenAPI specs
- Update example script to demonstrate edge case handling for API endpoint names
- Ensure proper session and connector closure in async API requests
- Add comprehensive test cases for problematic path name conversions
Addresses potential issues with API endpoint generation and async request management, improving overall robustness of API integration capabilities.

refactor(extractor): Enhance HTML processing and base64 image removal

- Consolidate BeautifulSoup operations for more robust HTML processing
- Implement comprehensive base64 image removal strategy with detailed debugging
- Add multiple removal techniques for base64 images in img tags, anchors, and styles
- Improve error handling and type checking during HTML manipulation
- Update example code to use different LLM model and async extraction method
- Add debug print statements to track base64 image removal process
- Refactor main content extraction and tag filtering logic

## v0.9.25

- refactor(extractor): reorganize imports and add model_config attribute

- Moved the import of run_sync to a more appropriate location
- Introduced model_config attribute using ConfigDict for better configuration management

refactor(whatsapp): streamline WhatsApp bot structure and introduce v2 components

- Removed unnecessary context_manager field from WhatsAppBot class.
- Updated AudioMessage class to improve type handling in convert_long_to_str method.
- Added new v2 module with BotConfig, BatchProcessorManager, and message limit definitions for enhanced configuration and processing capabilities.
- Introduced new files for in-memory batch processing and payload handling.
- Established a new WhatsAppBot class in v2 for better organization and functionality.

