# LangChain Tools Agent

An interactive LangChain agent powered by Groq. The agent can decide when to call registered tools while answering user requests.

## Features

- Groq chat model through `ChatGroq`
- LangChain tool calling with `@tool`
- Interactive command-line conversation
- CPU, memory, and disk metrics
- Host ping support on Windows and Linux/macOS
- Echo tool for returning text unchanged

## Installation

From the project root, install the required packages:

```powershell
python -m pip install -U langchain-core langchain-groq psutil
```

## API Key

Set your Groq API key in PowerShell for the current terminal session:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
```

Do not commit API keys to Git or place them directly in the source code.

## Run

From `C:\Yash\VAP`, run:

```powershell
python .\day4\langchain\tools_agent\langchain_agent.py
```

Type a request at the `You:` prompt. Type `exit` to stop the agent.

## Available Tools

- `echo(text)`: Returns the supplied text unchanged.
- `get_system_metrics()`: Returns CPU, memory, and disk usage as JSON.
- `ping_host(hostname)`: Pings a hostname and returns the command result as JSON.

## Adding a Tool

1. Define a function and decorate it with `@tool`.
2. Add the function to the `TOOLS` list.

Example:

```python
@tool
def current_directory() -> str:
    """Return the current working directory."""
    return os.getcwd()


TOOLS = [echo, get_system_metrics, ping_host, current_directory]
```

The agent automatically exposes registered tools to the Groq model and dispatches tool calls by name.

## Model

The agent currently uses:

```text
qwen/qwen3.8-27b
```

Change `MODEL_NAME` in `langchain_agent.py` if you want to use another model available in your Groq account.
