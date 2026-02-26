from langgraph.graph import StateGraph, MessagesState, START
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
import os
from weather_service.configuration import Configuration
from weather_service.observability import get_tracer

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
     final_answer: str = ""

def get_mcpclient():
    """Create an MCP client.

    Trace context propagation (traceparent headers) to the MCP gateway is
    handled automatically by opentelemetry-instrumentation-httpx, which
    injects the current span's context on every outgoing HTTP request.
    """
    return MultiServerMCPClient({
        "math": {
            "url": os.getenv("MCP_URL", "http://localhost:8000/mcp"),
            "transport": os.getenv("MCP_TRANSPORT", "streamable_http"),
        }
    })

async def get_graph(client) -> StateGraph:
    llm = ChatOpenAI(
        model=config.llm_model,
        openai_api_key=config.llm_api_key,
        openai_api_base=config.llm_api_base,
        temperature=0,
    )

    # Get tools asynchronously
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with providing weather information. You must use the provided tools to complete your task.")

    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        tracer = get_tracer()
        with tracer.start_as_current_span("gen_ai.chat", attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.system": "openai",
        }):
            result = llm_with_tools.invoke([sys_msg] + state["messages"])
        state["messages"].append(result)
        if isinstance(result, AIMessage) and not result.tool_calls:
            state["final_answer"] = result.content
        return state

    # Traced tool node: wraps each tool call in a gen_ai.tool span so that
    # the httpx POST /mcp request (and downstream mcp-router spans) nest under it.
    class TracedToolNode(ToolNode):
        def invoke(self, input, config=None, **kwargs):
            tracer = get_tracer()
            # Extract tool names from the pending tool calls in the last AI message
            messages = input.get("messages", [])
            tool_names = []
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    tool_names = [tc["name"] for tc in msg.tool_calls]
                    break
            span_name = f"gen_ai.tool ({', '.join(tool_names)})" if tool_names else "gen_ai.tool"
            with tracer.start_as_current_span(span_name, attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": ", ".join(tool_names),
            }):
                return super().invoke(input, config, **kwargs)

    # Build graph
    builder = StateGraph(ExtendedMessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", TracedToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    graph = builder.compile()
    return graph

# async def main():
#     from langchain_core.messages import HumanMessage
#     client = get_mcpclient()
#     graph = await get_graph(client)
#     messages = [HumanMessage(content="how is the weather in NY today?")]
#     async for event in graph.astream({"messages": messages}, stream_mode="updates"):
#         print(event)
#         output = event
#     output = output.get("assistant", {}).get("final_answer")
#     print(f">>> {output}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
