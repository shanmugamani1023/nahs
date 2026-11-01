# Day 23 — LangGraph

## Concepts Covered
- What is LangGraph
- Core concepts (Node, Edge, State, Graph)
- What is State
- Simple research agent example
- Code example
- LangGraph vs LangChain Agent
- Multi-agent with LangGraph
- When to use what

---

## 1. What is LangGraph?

LangChain agents work well for simple tasks. But what if your agent needs:
- Different paths based on conditions
- Multiple agents working together
- Loop back and retry on failure
- Track state across many steps

**LangGraph = build agents as a graph (nodes + edges) with full control over flow.**

**Analogy:** LangChain agent = a person who figures out steps as they go. LangGraph = a flowchart where you define every possible path, decision point, and loop in advance.

---

## 2. Core Concepts

```
Node  → a step (function that does something)
Edge  → connection between nodes
State → shared data bag passed between all nodes
Graph → all nodes + edges connected together
```

---

## 3. What is State?

**State = shared data bag that every node can read and update.**

Think of it like a form being passed between departments:

```
Form starts empty:
{
  question:      "What is LangGraph?"
  search_result: ""
  final_answer:  ""
  retry_count:   0
}
```

Node 1 (search) reads `question`, fills `search_result`:
```
{
  question:      "What is LangGraph?"
  search_result: "LangGraph is a library..."   ← filled
  final_answer:  ""
  retry_count:   0
}
```

Node 2 (check) reads `search_result`, updates retry:
```
{
  question:      "What is LangGraph?"
  search_result: "LangGraph is a library..."
  final_answer:  ""
  retry_count:   1                             ← updated
}
```

Node 3 (answer) reads everything, fills `final_answer`:
```
{
  question:      "What is LangGraph?"
  search_result: "LangGraph is a library..."
  final_answer:  "LangGraph is a tool for..." ← filled
  retry_count:   1
}
```

Without state:
```
Node 1 does search → result disappears
Node 2 has no idea what Node 1 found
```

With state:
```
Node 1 saves to state → Node 2 reads from state → Node 3 reads from state
All nodes share the same data bag
```

**Analogy:** State = patient file in a hospital. Every department (reception, doctor, lab, pharmacy) reads and writes to the same file. No one loses information between departments.

---

## 4. Flow Diagram — Research Agent

```
         ┌─────────┐
         │  START  │
         └────┬────┘
              ↓
         ┌─────────┐
         │  search │  ← Node 1
         └────┬────┘
              ↓
         ┌─────────┐
         │  check  │  ← Node 2 (is answer good?)
         └────┬────┘
              ↓
      ┌───────┴────────┐
      ↓                ↓
  [good]           [not good]
      ↓                ↓
 ┌────────┐      back to search ↑
 │ answer │
 └────────┘
```

---

## 5. Full Code Example

```python
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict

# ── 1. Define State ───────────────────────────────────────
class AgentState(TypedDict):
    question: str
    search_result: str
    final_answer: str
    retry_count: int

# ── 2. Define Nodes (functions) ───────────────────────────
llm = ChatAnthropic(model="claude-opus-4-6", api_key="your-key")

def search_node(state: AgentState) -> AgentState:
    """Search for information"""
    result = f"Search result for: {state['question']}"
    return {**state, "search_result": result}

def check_node(state: AgentState) -> AgentState:
    """Check if result is good enough"""
    response = llm.invoke(
        f"Is this a complete answer? Result: {state['search_result']}. "
        f"Reply only YES or NO."
    )
    return {**state, "check": response.content}

def answer_node(state: AgentState) -> AgentState:
    """Generate final answer"""
    response = llm.invoke(
        f"Question: {state['question']}\n"
        f"Context: {state['search_result']}\n"
        f"Give a clear answer."
    )
    return {**state, "final_answer": response.content}

# ── 3. Conditional Edge ───────────────────────────────────
def should_retry(state: AgentState) -> str:
    """Decide: answer or retry?"""
    if "YES" in state.get("check", "") or state["retry_count"] >= 2:
        return "answer"     # go to answer node
    else:
        return "search"     # go back to search

# ── 4. Build Graph ────────────────────────────────────────
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("search", search_node)
graph.add_node("check", check_node)
graph.add_node("answer", answer_node)

# Add edges
graph.set_entry_point("search")           # start here
graph.add_edge("search", "check")         # search → check
graph.add_conditional_edges(              # check → answer OR search
    "check",
    should_retry,
    {"answer": "answer", "search": "search"}
)
graph.add_edge("answer", END)             # answer → done

# ── 5. Run ────────────────────────────────────────────────
app = graph.compile()

result = app.invoke({
    "question": "What is LangGraph?",
    "search_result": "",
    "final_answer": "",
    "retry_count": 0
})

print(result["final_answer"])
```

---

## 6. LangGraph vs LangChain Agent

| | LangChain Agent | LangGraph |
|--|----------------|-----------|
| Flow | LLM decides dynamically | You define explicitly |
| Branching | Limited | Full control (if/else paths) |
| State | Basic | Full typed state object |
| Multi-agent | Hard | Built-in |
| Retry logic | Hard | Easy (loop back edge) |
| Debugging | Hard | Easy (see every node) |
| Use when | Simple tasks | Complex workflows |

---

## 7. Multi-Agent with LangGraph

Two agents working together:

```
┌──────────────┐     ┌──────────────┐
│  Researcher  │────→│   Writer     │
│  (searches)  │     │  (writes     │
│              │     │   report)    │
└──────────────┘     └──────────────┘
```

```python
def researcher_node(state):
    """Agent 1: finds information"""
    result = search_tool(state["topic"])
    return {**state, "research": result}

def writer_node(state):
    """Agent 2: writes based on research"""
    report = llm.invoke(f"Write a report on: {state['research']}")
    return {**state, "report": report.content}

graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)
```

---

## 8. When to Use What

```
Simple Q&A with tools     → LangChain Agent
Fixed pipeline (A→B→C)    → LangChain Chain (LCEL)
Complex branching logic    → LangGraph
Multi-agent workflows      → LangGraph
Retry on failure           → LangGraph
Human-in-the-loop          → LangGraph
```

---

## Summary

```
LangGraph        = agent as a graph (nodes + edges + state)
Node             = one step/function
Edge             = connection between steps
State            = shared data bag flowing through all nodes
Conditional edge = if/else branching between nodes

Advantage over LangChain Agent:
  Full control over flow
  Easy retry + branching
  Multi-agent support
  Typed state = easier debugging
```
