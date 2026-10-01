import ollama
MODEL = "qwen3:8b"



def echo(text: str) -> str:
    """Return the given text unchanged.
    Args:
        text: The text to echo back.
    """
    return text

def upper (text: str) -> str:
    """Return the given text in uppercase.
    Args:
        text: The text to convert to uppercase.
    """
    return text.upper()


TOOLS = [echo, upper]

SYSTEM_PROMPT = """
You are an echo agent.
When the user wants something repeated or echoed, call the echo tool with
exactly that text, then reply with the tool's result.
Never claim to have echoed anything unless the tool returned it.
"""


def run_agent():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    tool_map = {tool.__name__: tool for tool in TOOLS}

    print("Echo Agent (type 'exit' to quit)")

    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "exit":
            break
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})

        while True:
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            messages.append(response.message)

            if not response.message.tool_calls:
                print("\nAgent:", response.message.content)
                break

            for call in response.message.tool_calls:
                name = call.function.name
                args = call.function.arguments
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
                    {"role": "tool", "tool_name": name, "content": str(result)}
                )

if __name__ == "__main__":
    run_agent()