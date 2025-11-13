# Changelog

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
