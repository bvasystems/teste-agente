import asyncio
from agentle.agents.agent import Agent

async def test_typing():
    # Test non-streaming (should return AgentRunOutput)
    agent = Agent()
    
    # This should be typed as AgentRunOutput
    result = await agent.run_async("test", stream=False)
    print(f"Non-streaming result type: {type(result)}")
    
    # This should be typed as AsyncIterator[AgentRunOutput]
    stream_result = await agent.run_async("test", stream=True)
    print(f"Streaming result type: {type(stream_result)}")
    
    # This should work without type errors
    async for chunk in stream_result:
        print(f"Chunk type: {type(chunk)}")
        break

if __name__ == "__main__":
    asyncio.run(test_typing())