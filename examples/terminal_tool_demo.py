
# This example requires setting up an actual LLM provider, so we'll use a mocked interaction for demonstration.
# In a real scenario, you would use OpenRouter or another provider.

import asyncio
import os
import uuid
from datetime import datetime

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

# --- 1. Define the Terminal Tool ---
# The @terminal decorator marks this function as a terminal action.
# 'message_param="final_response"' tells the agent to use the 'final_response' argument 
# as the assistant's final output message in the chat history.
@terminal(message_param="final_response")
def submit_report(report_id: str, status: str, final_response: str) -> str:
    """Submits the final report and ends the conversation."""
    print(f"\n[Terminating Tool Executed] Report {report_id} submitted with status: {status}")
    return "Report submitted successfully."

# --- 2. Mock Provider (Simulating LLM behavior) ---
class DemoMockProvider(GenerationProvider):
    def __init__(self):
        self.step = 0

    async def generate_async(self, messages, tools=None, **kwargs):
        self.step += 1
        
        # Simulate LLM deciding to call the terminal tool
        if self.step == 1:
            print("\n[LLM] Deciding to call terminal tool...")
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
                                # LLM generates a tool call suggestion
                                ToolExecutionSuggestion(
                                    id="call_abc123",
                                    tool_name="submit_report",
                                    args={
                                        "report_id": "RPT-101",
                                        "status": "APPROVED",
                                        "final_response": "I have successfully submitted report RPT-101. The task is complete."
                                    }
                                )
                            ],
                            parsed=None
                        )
                    )
                ],
                usage=Usage(prompt_tokens=10, completion_tokens=10)
            )
        return Generation.mock() # Should not be reached

    async def stream_async(self, messages, tools=None, **kwargs):
        yield await self.generate_async(messages, tools, **kwargs)

    @property
    def default_model(self) -> str: return "mock"
    @property
    def map_model_kind_to_provider_model(self) -> dict: return {}
    @property
    def organization(self) -> str: return "mock"
    @property
    def price_per_million_tokens_input(self) -> float: return 0.0
    @property
    def price_per_million_tokens_output(self) -> float: return 0.0

# --- 3. Run the Agent ---
async def main():
    print("--- Starting Agent with Terminal Tool ---")
    
    agent = Agent(
        generation_provider=DemoMockProvider(),
        tools=[submit_report], # Register the decorated tool
        instructions="You are a helpful assistant."
    )

    # Execute agent
    output = await agent.run_async("Please submit report RPT-101.")

    print("\n--- Execution Finished ---")
    print(f"Total History Length: {len(output.context.message_history)}")
    
    # Check the last message in history
    last_msg = output.context.message_history[-1]
    if isinstance(last_msg, AssistantMessage):
        print(f"\n[Final Agent Response]: {last_msg.parts[0].text}")
        print("(This message was injected from the 'final_response' parameter of the terminal tool)")

if __name__ == "__main__":
    asyncio.run(main())
