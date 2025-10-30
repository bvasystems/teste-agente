# Changelog

## v0.9.27

- refactor(extractor): reorganize imports and add model_config attribute

- Moved the import of run_sync to a more appropriate location
- Introduced model_config attribute using ConfigDict for better configuration management

refactor(whatsapp): streamline WhatsApp bot structure and introduce v2 components

- Removed unnecessary context_manager field from WhatsAppBot class.
- Updated AudioMessage class to improve type handling in convert_long_to_str method.
- Added new v2 module with BotConfig, BatchProcessorManager, and message limit definitions for enhanced configuration and processing capabilities.
- Introduced new files for in-memory batch processing and payload handling.
- Established a new WhatsAppBot class in v2 for better organization and functionality.

