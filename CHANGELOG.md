# Changelog

## v0.8.25
feat(circuit breakers): new duckdb and mywsql implementations

## v0.8.24
fix(whatsapp): thinking tags not properlly removed

## v0.8.23
refactor: rename protocol classes to follow single underscore convention and to fix not defined class.

## v0.8.22
feat(file): enhance file validation with mime type and content checks

Add comprehensive validation for file attachments including:
- MIME type verification against official types
- Content integrity checks for various file types
- Optional dependency handling for specialized validation
- Better error handling and type safety

## v0.8.21

fix(whatsapp_bot): reorder message handling to check rate limits first

Move rate limit check before message processing to prevent spam. Also ensures session state is properly updated when rate limiting occurs.