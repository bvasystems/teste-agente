# Changelog

## v0.9.32

- refactor(agents): Simplify message storage in conversation store

- Remove `.to_assistant_message()` method calls when adding messages
- Directly store message objects in conversation store
- Affects multiple methods in `_stream_direct_response()` and `_stream_with_tools()`
- Reduces unnecessary method calls and simplifies message handling
