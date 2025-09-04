# Changelog

## v0.8.20

- fix(whatsapp): enhance race condition protection and batch timing logic

- Add double-check lock pattern to prevent duplicate batch processors
- Track last_message_added_at to improve batch timing calculations
- Reset batch timer when new messages arrive to existing batch
- Ensure processing_token is not None before creating processor
