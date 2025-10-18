"""
Validator to detect tool/function leakage in AI responses.

This validator detects when the AI accidentally exposes internal tool definitions,
function signatures, or hallucinates tool call syntax in its text responses.
"""

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from agentle.guardrails.core.output_guardrail_validator import OutputGuardrailValidator
from agentle.guardrails.core.guardrail_result import GuardrailResult, GuardrailAction


class ToolLeakageValidator(OutputGuardrailValidator):
    """
    Detects when AI responses leak internal tool definitions or hallucinate tool calls.

    This validator checks for:
    1. JSON-like tool call syntax in text responses
    2. Function signatures or definitions
    3. Tool parameter schemas
    4. Internal tool names that shouldn't be exposed to users

    Example violations:
    - {"tool_name": {"param": true}}
    - function get_weather(location: str) -> dict:
    - Tool: search_database, Parameters: {"query": "..."}
    """

    def __init__(
        self,
        priority: int = 25,
        enabled: bool = True,
        tools: Sequence[Callable[..., Any]] | None = None,
        tool_names: list[str] | None = None,
        block_on_detection: bool = True,
        redact_leakage: bool = True,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the tool leakage validator.

        Args:
            priority: Validator priority (lower runs first)
            enabled: Whether validator is enabled
            tools: List of tool callables (functions) to monitor. Names will be extracted automatically.
            tool_names: List of tool names to watch for (if None and tools not provided, uses generic patterns)
            block_on_detection: Whether to block responses with tool leakage
            redact_leakage: Whether to redact detected tool leakage
            config: Additional configuration
        """
        super().__init__(
            name="tool_leakage",
            priority=priority,
            enabled=enabled,
            config=config or {},
        )

        # Extract tool names from callables if provided
        if tools:
            self.tool_names = self._extract_tool_names(tools)
        else:
            self.tool_names = tool_names or []

        self.block_on_detection = block_on_detection
        self.redact_leakage = redact_leakage

        # Patterns to detect tool leakage
        self.patterns = [
            # JSON-like tool calls: {"tool_name": {...}}
            r'\{\s*["\']?\w+["\']?\s*:\s*\{[^}]*\}\s*\}',
            # Function definitions: def function_name(...) or function function_name(...)
            r"(?:def|function)\s+\w+\s*\([^)]*\)",
            # Tool call syntax: Tool: name, Parameters: {...}
            r"Tool\s*:\s*\w+\s*,?\s*Parameters?\s*:\s*\{",
            # Function signatures with types: function_name(param: type) -> type
            r"\w+\s*\([^)]*:\s*\w+[^)]*\)\s*->\s*\w+",
            # Explicit tool mentions: "calling tool X" or "using function Y"
            r'(?:calling|using|executing)\s+(?:tool|function)\s+["\']?\w+["\']?',
            # Parameter schemas: {"name": "string", "type": "..."}
            r'\{\s*["\']name["\']\s*:\s*["\']string["\']',
        ]

    def _extract_tool_names(self, tools: Sequence[Callable[..., Any]]) -> list[str]:
        """
        Extract tool names from callable functions.

        Args:
            tools: Sequence of callable functions

        Returns:
            List of extracted tool names
        """
        tool_names = []

        for tool in tools:
            # Try to get the function name
            if hasattr(tool, "__name__"):
                tool_names.append(tool.__name__)
            # Handle Tool objects from agentle
            elif hasattr(tool, "name"):
                tool_names.append(tool.name)
            # Handle bound methods
            elif hasattr(tool, "__func__") and hasattr(tool.__func__, "__name__"):
                tool_names.append(tool.__func__.__name__)

        return tool_names

    async def perform_validation(
        self, content: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """
        Validate output for tool leakage.

        Args:
            content: The output text to validate
            context: Optional context (can include tool_names)

        Returns:
            GuardrailResult with action and details
        """
        # Get tool names from context if available
        tool_names_to_check = self.tool_names
        if context and "tool_names" in context:
            tool_names_to_check = context["tool_names"]

        # Check for tool leakage
        violations = []
        confidence = 0.0

        # 1. Check for JSON-like tool call patterns
        json_matches = self._detect_json_tool_calls(content)
        if json_matches:
            violations.extend(json_matches)
            confidence = max(confidence, 0.9)

        # 2. Check for function definitions
        function_matches = self._detect_function_definitions(content)
        if function_matches:
            violations.extend(function_matches)
            confidence = max(confidence, 0.85)

        # 3. Check for explicit tool name mentions
        if tool_names_to_check:
            tool_name_matches = self._detect_tool_name_leakage(
                content, tool_names_to_check
            )
            if tool_name_matches:
                violations.extend(tool_name_matches)
                confidence = max(confidence, 0.95)

        # 4. Check for generic tool patterns
        pattern_matches = self._detect_generic_patterns(content)
        if pattern_matches:
            violations.extend(pattern_matches)
            confidence = max(confidence, 0.7)

        # Determine action based on violations
        if not violations:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                confidence=1.0,
                reason="No tool leakage detected",
                validator_name=self.name,
            )

        # Build violation message
        violation_summary = f"Detected {len(violations)} tool leakage pattern(s)"

        # Redact if enabled
        modified_content = None
        if self.redact_leakage:
            modified_content = self._redact_tool_leakage(content, violations)
            action = GuardrailAction.MODIFY
            reason = f"{violation_summary}. Content has been redacted."
        elif self.block_on_detection:
            action = GuardrailAction.BLOCK
            reason = (
                f"{violation_summary}. Response blocked to prevent information leakage."
            )
        else:
            action = GuardrailAction.WARN
            reason = f"{violation_summary}. Warning issued but content allowed."

        return GuardrailResult(
            action=action,
            confidence=confidence,
            reason=reason,
            validator_name=self.name,
            modified_content=modified_content,
            metadata={
                "violations": violations,
                "violation_count": len(violations),
                "tool_names_checked": tool_names_to_check,
            },
        )

    def _detect_json_tool_calls(self, content: str) -> list[dict[str, Any]]:
        """Detect JSON-like tool call syntax."""
        violations = []

        # Look for JSON objects that might be tool calls
        try:
            # Find potential JSON blocks
            json_pattern = r"\{[^{}]*\{[^{}]*\}[^{}]*\}"
            matches = re.finditer(json_pattern, content, re.DOTALL)

            for match in matches:
                json_str = match.group()
                try:
                    # Try to parse as JSON
                    parsed = json.loads(json_str)

                    # Check if it looks like a tool call
                    if isinstance(parsed, dict) and len(parsed) == 1:
                        key = list(parsed.keys())[0]
                        value = parsed[key]

                        # Tool calls typically have a single key with dict/list value
                        if isinstance(value, (dict, list)):
                            violations.append(
                                {
                                    "type": "json_tool_call",
                                    "pattern": json_str[:100],
                                    "position": match.start(),
                                }
                            )
                except json.JSONDecodeError:
                    # Not valid JSON, but might still look suspicious
                    if "tool" in json_str.lower() or "function" in json_str.lower():
                        violations.append(
                            {
                                "type": "json_like_tool_syntax",
                                "pattern": json_str[:100],
                                "position": match.start(),
                            }
                        )
        except Exception:
            pass

        return violations

    def _detect_function_definitions(self, content: str) -> list[dict[str, Any]]:
        """Detect function definition syntax."""
        violations = []

        # Python-style function definitions
        pattern = r"(?:def|async\s+def)\s+(\w+)\s*\([^)]*\)"
        matches = re.finditer(pattern, content)

        for match in matches:
            violations.append(
                {
                    "type": "function_definition",
                    "function_name": match.group(1),
                    "pattern": match.group()[:100],
                    "position": match.start(),
                }
            )

        return violations

    def _detect_tool_name_leakage(
        self, content: str, tool_names: list[str]
    ) -> list[dict[str, Any]]:
        """Detect explicit mentions of tool names in suspicious contexts."""
        violations = []

        for tool_name in tool_names:
            # Look for tool name in suspicious contexts
            suspicious_contexts = [
                rf"\b{re.escape(tool_name)}\s*\(",  # function call syntax
                rf'\{{\s*["\']?{re.escape(tool_name)}["\']?\s*:',  # JSON key
                rf'(?:tool|function)\s+["\']?{re.escape(tool_name)}["\']?',  # explicit mention
            ]

            for pattern in suspicious_contexts:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    violations.append(
                        {
                            "type": "tool_name_leakage",
                            "tool_name": tool_name,
                            "pattern": match.group()[:100],
                            "position": match.start(),
                        }
                    )

        return violations

    def _detect_generic_patterns(self, content: str) -> list[dict[str, Any]]:
        """Detect generic tool leakage patterns."""
        violations = []

        for pattern in self.patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                violations.append(
                    {
                        "type": "generic_pattern",
                        "pattern": match.group()[:100],
                        "position": match.start(),
                    }
                )

        return violations

    def _redact_tool_leakage(
        self, content: str, violations: list[dict[str, Any]]
    ) -> str:
        """Redact detected tool leakage from content."""
        # Sort violations by position (descending) to avoid offset issues
        sorted_violations = sorted(
            violations, key=lambda x: x.get("position", 0), reverse=True
        )

        modified_content = content

        for violation in sorted_violations:
            pattern = violation.get("pattern", "")
            if pattern and pattern in modified_content:
                # Replace with redacted message
                redaction = "[REDACTED: Internal tool information]"
                modified_content = modified_content.replace(pattern, redaction, 1)

        return modified_content
