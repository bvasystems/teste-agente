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

if TYPE_CHECKING:
    from agentle.agents.agent_input import AgentInput
    from agentle.generations.models.generation.trace_params import TraceParams


class ProgressiveProfilingAgent(BaseModel):
    """An agent specialized in progressive data collection using structured outputs"""

    field_specs: Sequence[FieldSpec]
    generation_provider: GenerationProvider
    model: str | None = None
    max_attempts_per_field: int = 3
    conversational: bool = True  # Whether to maintain conversational flow

    # Private attributes
    _agent: Agent[CollectedData] | None = PrivateAttr(default=None)
    _collected_data: MutableMapping[str, Any] = PrivateAttr(default_factory=dict)
    _attempts: MutableMapping[str, int] = PrivateAttr(default_factory=dict)
    _conversation_history: MutableSequence[str] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    @property
    def agent(self) -> Agent[CollectedData]:
        if self._agent is None:
            raise ValueError("ERROR: Agent is not set.")
        return self._agent

    def model_post_init(self, context: Any) -> None:
        """Initialize the internal agent with instructions for structured output"""
        # Build field descriptions for instructions
        field_descriptions = self._build_field_descriptions()

        # Create specialized instructions
        instructions = dedent(f"""\
        You are a friendly data collection specialist focused on progressive profiling.
        Your goal is to collect all required information from the user in a natural, conversational way.

        ## Fields to Collect:
        {field_descriptions}

        ## Guidelines:
        1. Be conversational and friendly while collecting information
        2. Ask for one or a few related fields at a time, not all at once
        3. Extract and validate any information the user provides
        4. If a user provides multiple pieces of information at once, extract all of them
        5. Be flexible - users might provide information in any order
        6. Handle corrections gracefully if users want to update previously provided information
        7. Provide a conversational response while collecting data

        ## Response Format:
        You MUST always return a CollectedData object with:
        - fields: A dictionary containing ALL collected data (both previous and new)
        - pending_fields: List of field names still needed (only required fields)
        - completed: Whether all required fields have been collected

        ## Current State:
        The current state of collected data and conversation history will be provided in the input.
        You should:
        1. Acknowledge what has already been collected
        2. Extract any new information from the user's message
        3. Merge new data with existing data in your response
        4. Update the pending_fields list accordingly
        5. Set completed=true only when all required fields are collected

        ## Validation Rules:
        - string: Any text value
        - integer: Must be a valid whole number
        - float: Must be a valid decimal number
        - boolean: Accept "yes", "no", "true", "false", "1", "0"
        - email: Must contain @ and a domain
        - date: Accept common date formats

        Always provide a natural, conversational response while ensuring the structured data is complete and accurate.
        """)

        # Create the internal agent with structured output only
        self._agent = Agent(
            name="Progressive Profiling Agent",
            description="An agent that progressively collects user information through conversation",
            generation_provider=self.generation_provider,
            model=self.model or self.generation_provider.default_model,
            instructions=instructions,
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
        # Build the current state context
        state_context = self._build_state_context(input)

        # Run the internal agent with state context
        result = self.agent.run(
            state_context, timeout=timeout, trace_params=trace_params
        )

        # Update internal state based on the response
        if result.parsed:
            self._update_state_from_response(result.parsed, input)

        return result

    async def run_async(
        self, input: AgentInput | Any, *, trace_params: TraceParams | None = None
    ) -> AgentRunOutput[CollectedData]:
        """Run the progressive profiling agent asynchronously"""
        # Build the current state context
        state_context = self._build_state_context(input)

        # Run the internal agent with state context
        result = await self.agent.run_async(state_context, trace_params=trace_params)

        # Update internal state based on the response
        if result.parsed:
            self._update_state_from_response(result.parsed, input)

        return result

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
        self._conversation_history.clear()

    def get_collected_data(self) -> Mapping[str, Any]:
        """Get the currently collected data"""
        return dict(self._collected_data)

    def is_complete(self) -> bool:
        """Check if all required fields have been collected"""
        return self._check_completion()

    def _build_state_context(self, user_input: Any) -> str:
        """Build the context including current state and user input"""
        # Current collected data
        collected_info = {
            field_name: value for field_name, value in self._collected_data.items()
        }

        # Pending fields
        pending_fields = self._get_pending_fields()

        # Build state summary
        state_summary = {
            "collected_data": collected_info,
            "pending_required_fields": pending_fields,
            "total_fields": len(self.field_specs),
            "required_fields": len([fs for fs in self.field_specs if fs.required]),
            "optional_fields": len([fs for fs in self.field_specs if not fs.required]),
        }

        # Build conversation history context (last 3 exchanges)
        history_context = ""
        if self._conversation_history:
            recent_history = self._conversation_history[-6:]  # Last 3 exchanges
            history_context = "\n\nRecent conversation:\n" + "\n".join(recent_history)

        # Combine everything
        context = dedent(f"""\
        ## Current State:
        {json.dumps(state_summary, indent=2)}
        {history_context}
        
        ## User Input:
        {user_input}
        
        Please analyze the user input, extract any relevant field values, and return the complete CollectedData object with all collected fields (both previous and new).
        """)

        return context

    def _update_state_from_response(
        self, response: CollectedData, user_input: Any
    ) -> None:
        """Update internal state based on the agent's response"""
        # Update collected data with any new fields
        for field_name, value in response.fields.items():
            # Validate that the field exists in our spec
            field_spec = next(
                (fs for fs in self.field_specs if fs.name == field_name), None
            )
            if field_spec:
                # Convert and validate the value
                try:
                    converted_value = self._convert_value(value, field_spec.type)
                    self._collected_data[field_name] = converted_value
                except ValueError:
                    # Skip invalid values
                    pass

        # Update conversation history
        if isinstance(user_input, str):
            self._conversation_history.append(f"User: {user_input}")

        # We don't store the agent's response text here since it's in the AgentRunOutput

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

    def get_collection_status(self) -> str:
        """Get a human-readable status of the collection progress"""
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
