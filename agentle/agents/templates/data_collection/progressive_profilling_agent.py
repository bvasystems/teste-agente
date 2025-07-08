from __future__ import annotations

import json
from collections.abc import (
    AsyncGenerator,
    Generator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.private_attr import PrivateAttr

from agentle.agents.agent import Agent
from agentle.agents.agent_run_output import AgentRunOutput
from agentle.agents.templates.data_collection.collected_data import CollectedData
from agentle.agents.templates.data_collection.field_spec import FieldSpec
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from agentle.agents.agent_input import AgentInput
    from agentle.generations.models.generation.trace_params import TraceParams


class ProgressiveProfilingAgent(BaseModel):
    """An agent specialized in progressive data collection"""

    field_specs: Sequence[FieldSpec]
    generation_provider: GenerationProvider
    model: str | None = None
    max_attempts_per_field: int = 3
    conversational: bool = True  # Whether to maintain conversational flow

    # Private attributes
    _agent: Agent[CollectedData] | None = PrivateAttr(default=None)
    _collected_data: MutableMapping[str, Any] = PrivateAttr(default_factory=dict)
    _attempts: MutableMapping[str, int] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    @property
    def agent(self) -> Agent[CollectedData]:
        if self._agent is None:
            raise ValueError("ERROR: Agent is not set.")
        return self._agent

    def model_post_init(self, context: Any) -> None:
        """Initialize the internal agent with appropriate tools and instructions"""
        # Create tools for data collection
        save_tool = Tool.from_callable(
            self._save_field,
        )

        list_tool = Tool.from_callable(
            self._list_collected_fields,
        )

        validate_tool = Tool.from_callable(
            self._validate_field,
        )

        get_state_tool = Tool.from_callable(
            self._get_current_state,
        )

        # Build field descriptions for instructions
        field_descriptions = self._build_field_descriptions()

        # Create specialized instructions
        instructions = dedent(f"""\
        You are a data collection specialist focused on progressive profiling.
        Your goal is to collect all required information from the user in a natural, conversational way.

        ## Fields to Collect:
        {field_descriptions}

        ## Guidelines:
        1. Be conversational and friendly while collecting information
        2. Ask for one or a few related fields at a time, not all at once
        3. Validate data before saving using the validate_field tool
        4. Use the save_field tool to store validated data
        5. Check progress with list_collected_fields tool
        6. Always call get_current_state at the end to get the current CollectedData state
        7. If a user provides multiple pieces of information at once, extract and save all of them
        8. Be flexible - users might provide information in any order
        9. Handle corrections gracefully if users want to update previously provided information

        ## IMPORTANT - Response Format:
        You MUST always return your response as a CollectedData object by calling get_current_state.
        Your response should include:
        - fields: The current collected data
        - pending_fields: List of field names still needed
        - completed: Whether all required fields have been collected

        Always structure your final response using the current state from get_current_state.
        """)

        # Create the internal agent
        self._agent = Agent(
            name="Progressive Profiling Agent",
            description="An agent that progressively collects user information",
            generation_provider=self.generation_provider,
            model=self.model or self.generation_provider.default_model,
            instructions=instructions,
            tools=[save_tool, list_tool, validate_tool, get_state_tool],
            response_schema=CollectedData,
        )

    def run(
        self,
        input: AgentInput | Any,
        *,
        timeout: float | None = None,
        trace_params: TraceParams | None = None,
    ) -> AgentRunOutput[CollectedData]:
        """Run the progressive profiling agent"""
        # Include current state in the input context
        current_state_info = (
            f"\n\nCurrent collected data: {json.dumps(self._collected_data, indent=2)}"
            if self._collected_data
            else ""
        )

        # Run the internal agent with state context
        enhanced_input = f"{input}{current_state_info}"
        return self.agent.run(
            enhanced_input, timeout=timeout, trace_params=trace_params
        )

    async def run_async(
        self, input: AgentInput | Any, *, trace_params: TraceParams | None = None
    ) -> AgentRunOutput[CollectedData]:
        """Run the progressive profiling agent asynchronously"""
        # Include current state in the input context
        current_state_info = (
            f"\n\nCurrent collected data: {json.dumps(self._collected_data, indent=2)}"
            if self._collected_data
            else ""
        )

        # Run the internal agent with state context
        enhanced_input = f"{input}{current_state_info}"
        return await self.agent.run_async(enhanced_input, trace_params=trace_params)

    @contextmanager
    def start_mcp_servers(self) -> Generator[None, None, None]:
        """Start MCP servers for the internal agent"""
        with self.agent.start_mcp_servers():
            yield

    @asynccontextmanager
    async def start_mcp_servers_async(self) -> AsyncGenerator[None, None]:
        """Start MCP servers asynchronously for the internal agent"""
        async with self.agent.start_mcp_servers_async():
            yield

    def reset(self) -> None:
        """Reset the collected data to start fresh"""
        self._collected_data.clear()
        self._attempts.clear()

    def get_collected_data(self) -> Mapping[str, Any]:
        """Get the currently collected data"""
        return dict(self._collected_data)

    def is_complete(self) -> bool:
        """Check if all required fields have been collected"""
        return self._check_completion()

    def _build_field_descriptions(self) -> str:
        """Build a formatted description of all fields to collect"""
        descriptions: MutableSequence[str] = []

        for spec in self.field_specs:
            desc = f"- **{spec.name}** ({spec.type})"
            if spec.required:
                desc += " [REQUIRED]"
            desc += f": {spec.description}"

            if spec.examples:
                desc += f"\n  Examples: {', '.join(spec.examples)}"

            if spec.validation:
                desc += f"\n  Validation: {spec.validation}"

            descriptions.append(desc)

        return "\n".join(descriptions)

    def _save_field(self, field_name: str, value: Any) -> str:
        """Save a collected field value"""
        # Find the field spec
        field_spec = next(
            (fs for fs in self.field_specs if fs.name == field_name), None
        )

        if not field_spec:
            return f"Error: Field '{field_name}' is not a recognized field."

        # Type conversion based on field spec
        try:
            converted_value = self._convert_value(value, field_spec.type)
            self._collected_data[field_name] = converted_value

            # Reset attempts for this field on successful save
            self._attempts[field_name] = 0

            # Return current state info
            pending = self._get_pending_fields()
            return f"Successfully saved {field_name}: {converted_value}\n\nCurrent state: {len(self._collected_data)} fields collected, {len(pending)} still needed."
        except ValueError as e:
            return f"Error saving {field_name}: {str(e)}"

    def _validate_field(self, field_name: str, value: Any) -> str:
        """Validate a field value against its specification"""
        field_spec = next(
            (fs for fs in self.field_specs if fs.name == field_name), None
        )

        if not field_spec:
            return f"Error: Field '{field_name}' is not recognized."

        try:
            # Type validation
            converted_value = self._convert_value(value, field_spec.type)

            # Custom validation if provided
            if field_spec.validation:
                # This could be enhanced with actual validation logic
                return f"Validation passed for {field_name}: {converted_value}"

            return f"Valid: {field_name} = {converted_value} ({field_spec.type})"
        except ValueError as e:
            return f"Validation failed for {field_name}: {str(e)}"

    def _list_collected_fields(self) -> str:
        """List all collected fields and what's still needed"""
        collected: MutableSequence[str] = []
        pending: MutableSequence[str] = []

        for spec in self.field_specs:
            if spec.name in self._collected_data:
                collected.append(f"✓ {spec.name}: {self._collected_data[spec.name]}")
            elif spec.required:
                pending.append(f"○ {spec.name} ({spec.type}) - {spec.description}")
            else:
                pending.append(
                    f"○ {spec.name} ({spec.type}) [optional] - {spec.description}"
                )

        result = "## Collection Status:\n\n"

        if collected:
            result += "### Collected:\n" + "\n".join(collected) + "\n\n"

        if pending:
            result += "### Still Needed:\n" + "\n".join(pending)
        else:
            result += "### All required fields have been collected! ✓"

        return result

    def _get_current_state(self) -> CollectedData:
        """Get the current state as a CollectedData object"""
        return CollectedData(
            fields=dict(self._collected_data),
            pending_fields=self._get_pending_fields(),
            completed=self._check_completion(),
        )

    def _convert_value(self, value: Any, field_type: str) -> Any:
        """Convert a value to the specified type"""
        if field_type == "string":
            return str(value)
        elif field_type == "integer":
            return int(value)
        elif field_type == "float":
            return float(value)
        elif field_type == "boolean":
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1", "on")
            return bool(value)
        elif field_type == "email":
            # Basic email validation
            email_str = str(value).strip().lower()
            if "@" not in email_str or "." not in email_str.split("@")[1]:
                raise ValueError("Invalid email format")
            return email_str
        elif field_type == "date":
            # Could use dateutil.parser here for more robust parsing
            return str(value)  # Simplified for now
        else:
            return value

    def _check_completion(self) -> bool:
        """Check if all required fields have been collected"""
        for spec in self.field_specs:
            if spec.required and spec.name not in self._collected_data:
                return False
        return True

    def _get_pending_fields(self) -> MutableSequence[str]:
        """Get list of pending required fields"""
        pending: MutableSequence[str] = []
        for spec in self.field_specs:
            if spec.required and spec.name not in self._collected_data:
                pending.append(spec.name)
        return pending


if __name__ == "__main__":
    from agentle.generations.providers.google.google_generation_provider import (
        GoogleGenerationProvider,
    )

    # Define the fields to collect
    user_profile_fields = [
        FieldSpec(
            name="full_name",
            type="string",
            description="User's full name",
            examples=["John Doe", "Jane Smith"],
        ),
        FieldSpec(
            name="email",
            type="email",
            description="User's email address",
            validation="Must be a valid email format",
        ),
        FieldSpec(
            name="age",
            type="integer",
            description="User's age",
            validation="Must be between 13 and 120",
        ),
        FieldSpec(
            name="interests",
            type="string",
            description="User's interests or hobbies",
            required=False,
            examples=["reading", "sports", "cooking"],
        ),
    ]

    # Create the progressive profiling agent
    profiler = ProgressiveProfilingAgent(
        field_specs=user_profile_fields,
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        conversational=True,
    )

    # Start collecting data
    response = profiler.run("Hi! I'd like to sign up for your service.")
    print(response.text)
    exit()

    # Continue the conversation
    response = profiler.run("My name is John Doe")
    print(response.text)

    # Check progress
    if response.parsed:
        print(f"Collected: {response.parsed.fields}")
        print(f"Still needed: {response.parsed.pending_fields}")
        print(f"Complete: {response.parsed.completed}")

    # Continue until all fields are collected
    while not profiler.is_complete():
        user_input = input("You: ")
        response = profiler.run(user_input)
        print(f"Agent: {response.text}")

    # Get final collected data
    final_data = profiler.get_collected_data()
    print(f"Profile complete! Collected data: {final_data}")
