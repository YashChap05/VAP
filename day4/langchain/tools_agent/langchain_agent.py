import json
import os
import platform
import subprocess

import psutil
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

MODEL_NAME = "qwen/qwen3.8-27b"


@tool
def echo(text: str) -> str:
    """Return the given text unchanged."""
    return text


@tool
def get_system_metrics() -> str:
    """Get current CPU, memory, and disk usage."""
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage(os.path.abspath(os.sep)).percent,
    }
    return json.dumps(metrics)


@tool
def ping_host(hostname: str) -> str:
    """Ping a hostname and return the result."""
    count_flag = "-n" if platform.system() == "Windows" else "-c"
    try:
        result = subprocess.check_output(
            ["ping", count_flag, "4", hostname],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return json.dumps({"status": "success", "raw_output": result.strip()})
    except Exception as error:
        return json.dumps({"status": "error", "message": str(error)})


TOOLS = [echo, get_system_metrics, ping_host]
TOOL_MAP = {tool.name: tool for tool in TOOLS}

llm = ChatGroq(model=MODEL_NAME, temperature=0).bind_tools(TOOLS)
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful agent. Use the available tools when they are useful."
)


def run_agent() -> None:
    messages = [SYSTEM_MESSAGE]
    print("LangChain agent (type 'exit' to quit)")

    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        messages.append(HumanMessage(content=prompt))

        while True:
            response = llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                print("\nAgent:", response.content)
                break

            for tool_call in response.tool_calls:
                tool = TOOL_MAP.get(tool_call["name"])
                if tool is None:
                    result = "Error: Unknown tool."
                else:
                    try:
                        result = tool.invoke(tool_call["args"])
                    except Exception as error:
                        result = f"Error: {error}"

                print(f"\n[Tool: {tool_call['name']}] {result}")
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call["id"])
                )


if __name__ == "__main__":
    run_agent()
