import json
import os
import inspect
import subprocess
from typing import get_type_hints

import psutil
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "qwen/qwen3.8-27b"
def get_system_metrics():
    """Get system metrics such as CPU and memory usage."""
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
    }
    return json.dumps(metrics)


def echo(text: str) -> str:
    """Return the given text unchanged.
    Args:
        text: The text to echo back.
    """
    return text

def ping_host(hostname: str) -> str:
    """Ping a given hostname and return the result.
    Args:
        hostname: The hostname to ping.
    """
    try:
        res = subprocess.check_output(["ping", "-c", "4", hostname], stderr=subprocess.STDOUT, text=True)
        return json.dumps({"status": "success", "raw_output": res.strip()})
    except Exception as err:
        return json.dumps({"status": "error", "message": str(err)})

TOOL_FUNCTIONS = [echo, get_system_metrics, ping_host]


def build_tool_schema(tool):
    type_names = {str: "string", int: "integer", float: "number", bool: "boolean"}
    hints = get_type_hints(tool)
    properties = {}
    required = []

    for parameter in inspect.signature(tool).parameters.values():
        properties[parameter.name] = {
            "type": type_names.get(hints.get(parameter.name), "string")
        }
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)

    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": (inspect.getdoc(tool) or tool.__name__).split("\n")[0],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


TOOLS = [build_tool_schema(tool) for tool in TOOL_FUNCTIONS]
SYSTEM_PROMPT = """
You are a helpful agent. Use the available tools when they are useful.
"""


def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    tool_map = {tool.__name__: tool for tool in TOOL_FUNCTIONS}

    print("Agent (type 'exit' to quit)")

    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})

        while True:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                print("\nAgent:", message.content)
                break

            for call in message.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                tool = tool_map.get(name)

                if tool is None:
                    result = "Error: Unknown tool."
                else:
                    try:
                        result = tool(**args)
                    except Exception as exc:
                        result = f"Error: {exc}"
                
                print(f"\n[Tool: {name}] {result}")
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "name": name, "content": str(result)}
                 )
                
if __name__ == "__main__":
    run_agent()
