# Changelog

## v0.7.33
feat(context): add replace_message_history method to Context class

refactor(agent): improve conversation history handling with replace_message_history
Add support for keeping developer messages when replacing history and ensure proper message persistence in conversation store.

refactor(google-tool-adapter): enhance type parsing and schema generation
Add comprehensive type parser for complex Python type annotations and improve JSON Schema conversion with better handling of unions, optionals, and nested structures.

chore(examples): simplify open_memory.py by removing unused imports and server setup

## v0.7.32

feat(agent): improve tool call tracking and message history handling

- Replace called_tools with all_tool_suggestions and all_tool_results for better tracking across iterations
- Modify message history handling to ensure all tool suggestions and results are properly included
- Add append_tool_calls method to generation model for final response consolidation

## v0.7.31

refactor(agent): simplify tool execution results handling in message history

Improve message history handling by directly adding tool execution results to the last user message instead of complex multi-step processing. This makes the code more straightforward and reduces edge cases while maintaining the same functionality.

## v0.7.30

feat(agent): add change_instructions method to modify agent instructions

## v0.7.29
- feat(agent): add methods to modify agent properties

Add change_name, change_apis and change_endpoints methods to allow dynamic modification of agent properties

## v0.7.28

- feat(agent): add append_instructions method and improve tool execution handling

- Implement append_instructions to allow dynamic instruction updates
- Enhance tool execution flow by properly handling tool results in message history
