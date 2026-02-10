import uuid
from datetime import datetime
import pytest

from agentle.agents.agent import Agent
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.messages.generated_assistant_message import GeneratedAssistantMessage
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import ToolExecutionSuggestion
from agentle.generations.tools.terminal import terminal

# Mock Generation Provider
class MockGenerationProvider(GenerationProvider):
    def __init__(self):
        self.call_count = 0

    async def generate_async(self, messages, tools=None, **kwargs):
        self.call_count += 1
        
        if self.call_count == 1:
            # First call: Suggest the terminal tool
            return Generation(
                id=uuid.uuid4(),
                object="chat.generation",
                created=datetime.now(),
                model="mock-model",
                choices=[
                    Choice(
                        index=0,
                        message=GeneratedAssistantMessage(
                            parts=[
                                ToolExecutionSuggestion(
                                    id="call_1",
                                    tool_name="terminate_me",
                                    args={"reason": "Test termination", "msg": "Goodbye world"}
                                )
                            ],
                            parsed=None
                        )
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=10)
            )
        else:
            # Should not be reached if terminal works
            return Generation(
                id=uuid.uuid4(),
                object="chat.generation",
                created=datetime.now(),
                model="mock-model",
                choices=[
                    Choice(
                        index=0,
                        message=GeneratedAssistantMessage(
                            parts=[TextPart(text="Failed to terminate")],
                            parsed=None
                        )
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=10)
            )

    async def stream_async(self, messages, tools=None, **kwargs):
        # reuse generate logic for simplicity in mock
        yield await self.generate_async(messages, tools, **kwargs)

    @property
    def default_model(self) -> str:
        return "mock-model"

    @property
    def map_model_kind_to_provider_model(self) -> dict[str, str]:
        return {}

    @property
    def organization(self) -> str:
        return "mock-org"

    @property
    def price_per_million_tokens_input(self) -> float:
        return 0.0

    @property
    def price_per_million_tokens_output(self) -> float:
        return 0.0

# Terminal Tool
@terminal(message_param="msg")
def terminate_me(reason: str, msg: str) -> str:
    return "Tool executed"

@pytest.mark.asyncio
async def test_terminal_tool_stops_execution():
    agent = Agent(
        generation_provider=MockGenerationProvider(),
        tools=[terminate_me],
        instructions="Test agent"
    )
    
    # Run agent
    output = await agent.run_async("Start")
    
    # Verify context
    history = output.context.message_history
    
    # Expect:
    # 1. Developer message
    # 2. User message ("Start")
    # 3. Assistant message (Tool Call)
    # 4. User message (Tool Result)
    # 5. Assistant message (Injected from terminal param)
    
    assert len(history) == 5
    
    last_message = history[-1]
    assert isinstance(last_message, AssistantMessage)
    assert last_message.parts[0].text == "Goodbye world"
