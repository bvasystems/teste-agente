# Implementation Plan

- [x] 1. Create PricingService infrastructure
  - Create `agentle/responses/pricing/pricing_service.py` with abstract PricingService class
  - Create `agentle/responses/pricing/default_pricing_service.py` with static pricing dictionary
  - Create `agentle/responses/pricing/__init__.py` to export classes
  - _Requirements: 2.2, 2.3_

- [x] 2. Implement tracing helper methods in Responder
- [x] 2.1 Implement `_prepare_trace_input_data` method
  - Extract relevant fields from request_payload (input, model, tools, reasoning, etc.)
  - Return structured dictionary for trace context
  - _Requirements: 5.1, 5.3, 5.4_

- [x] 2.2 Implement `_prepare_trace_metadata` method
  - Extract model, provider, base_url
  - Merge custom metadata from generation_config
  - Return metadata dictionary
  - _Requirements: 5.1, 5.2_

- [x] 2.3 Implement `_extract_usage_from_response` method
  - Extract token usage from Response object
  - Handle missing usage data gracefully
  - Return usage dictionary with input/output/total tokens
  - _Requirements: 2.1, 2.5_

- [x] 2.4 Implement `_calculate_costs` method
  - Use PricingService to get model pricing
  - Calculate input and output costs
  - Return cost dictionary with breakdown
  - Handle unknown models gracefully
  - _Requirements: 2.2, 2.3, 2.4_

- [x] 3. Implement tracing context management methods
- [x] 3.1 Implement `_create_tracing_contexts` method
  - Iterate through all otel_clients
  - Create trace context for each client
  - Create generation context for each client
  - Store contexts in list of dictionaries
  - Handle context creation errors gracefully
  - _Requirements: 1.2, 1.5, 4.1, 4.2, 4.3_

- [x] 3.2 Implement `_update_tracing_success` method
  - Extract usage from response
  - Calculate costs using PricingService
  - Update generation contexts with output, usage, and costs
  - Update trace contexts with success status and metadata
  - Handle structured output parsing
  - Use fire-and-forget for non-critical operations
  - _Requirements: 1.3, 2.1, 2.2, 2.3, 2.4, 4.4, 5.2, 5.3, 5.5_

- [x] 3.3 Implement `_update_tracing_error` method
  - Update generation contexts with error information
  - Update trace contexts with failure status
  - Handle errors in error handling gracefully
  - _Requirements: 1.4, 4.1, 4.2, 4.3_

- [x] 3.4 Implement `_cleanup_tracing_contexts` method
  - Close all generation context generators
  - Close all trace context generators
  - Handle cleanup errors gracefully
  - _Requirements: 3.3, 4.5_

- [x] 4. Integrate tracing into non-streaming flow
- [x] 4.1 Modify `_respond_async` method for non-streaming
  - Add start_time tracking
  - Initialize active_contexts list
  - Call `_create_tracing_contexts` if otel_clients present
  - Wrap API call in try-except-finally
  - Call `_update_tracing_success` on success
  - Call `_update_tracing_error` on error
  - Call `_cleanup_tracing_contexts` in finally block
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.5, 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 5. Integrate tracing into streaming flow
- [x] 5.1 Create `_stream_events_with_tracing` wrapper method
  - Accept content_lines, text_format, active_contexts, start_time, model
  - Wrap `_stream_events_from_buffer` generator
  - Accumulate text deltas for metrics
  - Track final ResponseCompletedEvent
  - Call `_update_tracing_success` on completion
  - Call `_update_tracing_error` on error
  - _Requirements: 3.1, 3.2, 3.3, 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 5.2 Modify `_respond_async` method for streaming
  - Create tracing contexts before streaming
  - Pass contexts to `_stream_events_with_tracing`
  - Ensure cleanup happens after streaming completes
  - _Requirements: 3.1, 3.3, 4.5_

- [x] 6. Add PricingService to Responder initialization
- [x] 6.1 Add `pricing_service` parameter to `__init__`
  - Make it optional with default DefaultPricingService
  - Store as instance attribute
  - _Requirements: 2.2, 2.3_

- [x] 6.2 Update `openrouter` and `openai` class methods
  - Pass pricing_service parameter through
  - _Requirements: 2.2, 2.3_

- [x] 7. Add logging and error handling
- [x] 7.1 Add debug logging for tracing operations
  - Log context creation
  - Log context updates
  - Log context cleanup
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 7.2 Add error logging for tracing failures
  - Log context creation failures
  - Log update failures
  - Log cleanup failures
  - Ensure errors don't propagate to caller
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 8. Update type hints and imports
- [x] 8.1 Add necessary imports
  - Import datetime
  - Import fire_and_forget from rsb.coroutines
  - Import OtelClient types
  - Import PricingService
  - _Requirements: All_

- [x] 8.2 Update type hints
  - Add type hints to all new methods
  - Ensure consistency with existing code style
  - _Requirements: All_

- [x] 9. Documentation and examples
- [x] 9.1 Add docstrings to all new methods
  - Document parameters
  - Document return values
  - Document error handling behavior
  - _Requirements: All_

- [x] 9.2 Update Responder class docstring
  - Document otel_clients parameter
  - Document pricing_service parameter
  - Document tracing behavior
  - _Requirements: All_

- [x] 9.3 Create usage example
  - Create example showing Responder with OtelClient
  - Show both streaming and non-streaming usage
  - Demonstrate cost tracking
  - _Requirements: All_
