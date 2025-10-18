# Changelog

## v0.9.0 🚀

**Release Date:** October 18, 2024

We're thrilled to announce **Agentle v0.9.0**, a major milestone packed with powerful new features, critical fixes, and performance improvements. This release represents months of refinement across our parsing, provider, and agent ecosystems.

---

### ✨ **Highlights**

#### 🛡️ **Guardrails System**
- **NEW:** Comprehensive guardrails integration for safer AI interactions
- **Tool Leakage Validator:** Prevent sensitive tool information from leaking in responses

#### 🔌 **OpenRouter Provider**
- **Full OpenRouter integration** with streaming support and structured outputs
- **Dynamic price fetching** for cost-aware model selection
- **Model fallbacks** for improved reliability
- **Factory methods** for easier instantiation
- Enhanced tool adapter with complex type expansion in JSON Schema conversion
- Configurable httpx client timeout for generation tasks

#### 📄 **Document Parsing Revolution**
- **Native PDF processing** with PyMuPDF for superior performance
- **RTF Document Parser** - brand new format support
- **Native DOCX processing** option for faster document handling
- Enhanced PDF parsing with encrypted file support and structured exception handling
- PPTX parsing improvements with legacy conversion support
- Legacy `.doc` file conversion with improved error handling
- Optimized archive processing (ZIP, RAR) with better error messaging
- Page screenshot optimization to reduce per-image analysis costs
- Configurable `render_scale` for PDF screenshot quality control
- `max_concurrent_pages` field for PDF parsing concurrency control

#### 🧠 **Embeddings & Vector Stores**
- **DeepInfra Embedding Provider** - OpenAI-compatible API integration
- Batch embedding generation methods for async processing
- Enhanced async embedding methods with None value handling
- Load balancer implementation (DuckDB and in-memory) with provider quotas and ranking

#### 💬 **WhatsApp Enhancements**
- Enhanced markdown formatting (headers, tables, blockquotes, horizontal rules)
- Improved message splitting with list item grouping
- `send_message` method for independent message sending
- Multiple callback support with semantic type aliases
- Better table formatting (vertical lists for improved readability)

#### 🔧 **Developer Experience**
- **Circuit breaker** implementation with DuckDB backend
- Static knowledge `from_text` class method for easier content creation
- `append_content` method for dynamic ParsedFile content addition
- Provider ID implementation across GenerationProvider subclasses
- Fallback models parameter for provider resilience
- Enhanced file validation with MIME type and content checks

---

### 🐛 **Bug Fixes**

#### WhatsApp
- Fixed Brazilian phone number formatting after Meta's privacy update
- Proper handling of Evolution API typed dicts (converted to BaseModels)
- Fixed `from_number` extraction from sender
- Corrected buffer handling for Evolution API fields
- Improved thinking tag removal logic
- Enhanced race condition protection and batch timing
- Rate limit checks now happen before message processing

#### Agents & Tools
- Fixed assistant message context handling before tool processing
- Removed unnecessary cast in `_stream_with_tools` function
- Corrected property "text" update logic
- Fixed duplicate tools in vector stores
- Resolved duplicated parameter issues

#### Parsing & Generation
- Fixed PDF extraction error handling
- Handled cases where no extraction is returned from PDF provider
- Updated PyMuPDF import for better compatibility
- Normalized model names (removed 'google/' prefix)
- Fixed pickling errors in RAG search by avoiding callable serialization
- Improved logging calls

---

### ⚡ **Performance Improvements**

- **Single page screenshot per page** when images are present (avoiding duplicate OCR)
- Optimized PDF page processing with PyMuPDF
- Eliminated redundant temp file creation in archive processing
- Better markdown parsing performance
- Fail-fast on DOCX→PDF conversion errors

---

### 🔄 **Refactoring & Code Quality**

- Converted Evolution API typed dicts to BaseModels for better type safety
- Renamed 'agent' parameters to 'provider' for consistency
- Renamed 'chunk_tokens' to 'output_tokens' with new 'input_tokens' property
- Protocol classes follow single underscore convention
- Cleaned up whitespace and improved exception handling
- Streamlined parser class retrieval
- Made Langfuse a parameter for better flexibility
- Enhanced CallbackWithContext with token propagation

---

### 📚 **Documentation & Testing**

- Added test for uppercase file extension handling
- Testing failover Generation Provider
- Updated examples with lightweight implementations
- Comprehensive changelog updates

---

### 🙏 **Thank You**

This release wouldn't be possible without the dedication of our contributors and the valuable feedback from our community. We're committed to making Agentle the most powerful and reliable agentic AI framework.

**Upgrade now and experience the future of AI agents!**

```bash
pip install --upgrade agentle
```

---

## v0.8.68

- feat(openrouter) streaming with structured outputs
