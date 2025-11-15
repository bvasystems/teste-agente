# Changelog

## v0.9.37
fix(whatsapp): Add base64 media handling to avoid unnecessary downloads

- Include base64_data field in message dictionary conversion for media messages
- Check for existing base64 data before attempting media downloads in batch conversion
- Check for existing base64 data before attempting media downloads in single conversion
- Add base64_data parameter to WhatsAppAudioMessage creation in Evolution parser
- Add logging to indicate when base64 data is used vs. when downloads are needed
- Prevents redundant download attempts when media is already available in webhook payload
- Improves performance for audio messages that arrive with embedded base64 data

## v0.9.36
feat(openrouter): Enhance message adapter for tool execution and multimodal support

- Add support for converting tool execution results to OpenRouter messages
- Improve handling of messages with mixed content types (text, tool suggestions, tool results)
- Implement serialization methods for tool arguments and results
- Refactor message conversion logic to handle complex message scenarios
- Ensure proper separation and conversion of text, tool calls, and tool results
- Add support for preserving reasoning in assistant messages

## v0.9.35
refactor(whatsapp): Improve message splitting with enhanced line break preservation

- Completely refactored `_split_message_by_line_breaks` method
- Added more robust handling of paragraph and line breaks
- Preserved line breaks and formatting for lists and paragraphs
- Implemented smarter chunking of messages to maintain readability
- Added checks to keep entire messages intact when possible
- Improved handling of long lines and paragraphs
- Enhanced message splitting logic to maintain original text structure

## v0.9.34
fix(whatsapp): Preserve line breaks when splitting long messages

- Modify message splitting logic to retain original line breaks
- Remove `.strip()` calls to prevent unintended whitespace removal
- Ensure long messages maintain their original formatting and structure
- Prevents potential loss of formatting in multi-line WhatsApp messages

## v0.9.33
refactor(whatsapp): Improve message splitting and list handling for WhatsApp messages

- Enhance markdown formatting preservation in message processing
- Improve list detection and grouping logic to maintain formatting
- Modify message splitting to better handle paragraphs and lists
- Add more robust handling of line breaks and indentation
- Reduce list detection threshold to capture more complex list formats
- Prevent message fragmentation for list-based content
- Add null check for remote JID to prevent potential errors

## v0.9.32

- refactor(agents): Simplify message storage in conversation store

- Remove `.to_assistant_message()` method calls when adding messages
- Directly store message objects in conversation store
- Affects multiple methods in `_stream_direct_response()` and `_stream_with_tools()`
- Reduces unnecessary method calls and simplifies message handling
