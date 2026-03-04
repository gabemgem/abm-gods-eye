# abm-gods-eye

A drop-in LLM observer SDK for Python agent-based models.

Give any ABM a "god's eye" — an AI that can see the entire simulation and answer natural-language questions about what's happening, why, and what might come next.

## Architecture

```
┌─────────────────────────────────────────┐
│           abm-gods-eye SDK              │
│                                         │
│  ┌─────────────┐    ┌────────────────┐  │
│  │  LangGraph  │    │ LangChain Tools│  │
│  │  ReAct Agent│    │ (get_state,    │  │
│  │             │    │  query_agents, │  │
│  └──────┬──────┘    │  step_sim, ...) │  │
│         └──────────┬────────────────┘  │
│              ┌─────▼──────┐             │
│              │SimObserver │             │
│              │  Protocol  │             │
│              └─────┬──────┘             │
└────────────────────┼────────────────────┘
                     │  (your adapter)
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Mesa ABM    Agentpy ABM   Custom ABM
```

## Quickstart

```bash
uv add abm-gods-eye
export ANTHROPIC_API_KEY=sk-...
```

Implement the five-method `SimulationAdapter` protocol for your model, then:

```python
from abm_gods_eye import GodsEye
from my_sim import MyAdapter

eye = GodsEye(MyAdapter(my_model))

print(eye.ask("What patterns are emerging?"))
print(eye.ask("Step forward 10 steps and describe what changed."))
```

## The Adapter Protocol

The only thing you need to write is a thin wrapper around your ABM:

```python
class MyAdapter:
    def get_state(self) -> dict:       # full snapshot of current state
    def get_agents(self) -> list[dict]: # list of agent attribute dicts
    def get_metrics(self) -> dict:     # aggregate statistics
    def step(self, n: int = 1) -> None: # advance simulation n steps
    def get_history(self) -> list[dict]: # list of past state snapshots
```

See [examples/mesa_adapter.py](examples/mesa_adapter.py) for a complete working example with Mesa's Boltzmann Wealth Model.

## Tools Available to the LLM

| Tool | Description |
|------|-------------|
| `get_state` | Full simulation snapshot |
| `get_metrics` | Aggregate statistics |
| `query_agents` | Filter/inspect agents by attribute |
| `step_simulation` | Advance N steps and report changes |
| `get_history` | Past state snapshots (last 20) |
| `compare_states` | Delta between two historical states |

## Swapping LLM Backends

The default backend is Claude via `langchain-anthropic`. Pass any LangChain `BaseChatModel`:

```python
from langchain_openai import ChatOpenAI

eye = GodsEye(adapter, llm=ChatOpenAI(model="gpt-4o"))
```

## Development

```bash
git clone https://github.com/gabemgem/abm-gods-eye
cd abm-gods-eye
uv sync
uv run python examples/mesa_adapter.py
```
