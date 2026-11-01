# Day 22 — Agents + Tool Calling (ReAct Loop)

## Concepts Covered
- What is an AI Agent
- ReAct loop (Reasoning + Acting)
- What is Reasoning
- Tools — what agents can do
- How tool calling works
- LangChain Agent code
- Agent vs Chain
- RAG + Agent combo
- LangChain summary

---

## 1. What is an AI Agent?

Normal LLM: You ask → it answers. One shot. Done.

**Agent:** LLM that can **think → act → observe → repeat** until task is complete.

**Analogy:** Normal LLM = answering a quiz from memory. Agent = a detective who can go investigate, check evidence, make calls, come back, then give the answer.

---

## 2. ReAct Loop — How Agents Think

**ReAct = Reasoning + Acting**

```
Thought:     "I need to find today's weather in Chennai"
Action:      search_web("Chennai weather today")
Observation: "32°C, sunny"

Thought:     "Now I have the answer"
Action:      finish("Chennai is 32°C and sunny today")
```

Loop:
```
┌─────────────────────────────────┐
│  Thought  → what should I do?  │
│  Action   → call a tool        │
│  Observe  → get tool result    │
│  Repeat until done             │
└─────────────────────────────────┘
```

---

## 3. What is Reasoning?

**Reasoning = the "Thought" step — LLM thinks out loud before taking action.**

Without reasoning (direct answer):
```
User:  "What is 15% of 3847 in Euros?"
LLM:   "531.89 EUR"   ← just guesses, may hallucinate
```

With reasoning (think first, then act):
```
User: "What is 15% of 3847 in Euros?"

Thought: "I need to first calculate 15% of 3847,
          then convert USD to EUR.
          I should use calculator first."

Action: calculator("3847 * 0.15")
Result: 577.05

Thought: "Now I need EUR conversion."
Action: search_web("577.05 USD to EUR")
Result: 531.89 EUR

Answer: "531.89 EUR"
```

Why reasoning matters:
```
No reasoning  → LLM jumps to answer → wrong tool, wrong order, hallucination
With reasoning → LLM plans first   → correct tool, correct order, accurate answer
```

**Analogy:** You ask "how do I get to the airport?"
- Without reasoning → they immediately say "turn left" (might be wrong)
- With reasoning → they think "where are we? fastest route? traffic?" — THEN give directions.

---

## 4. Tools — What Agent Can Do

Tools = functions the LLM can call.

```python
tools = [
    search_web,       # search internet
    calculator,       # do math
    read_file,        # read a file
    run_python,       # execute code
    query_database,   # query SQL
    send_email,       # send email
]
```

LLM decides **which tool to call** and **with what input**, based on the task.

---

## 5. How Tool Calling Works

```
User: "What is 15% of 3847, and what is that in Euros?"

LLM thinks:
  Step 1 → need to calculate 15% of 3847
  Step 2 → need currency conversion

LLM calls:  calculator("3847 * 0.15")
Result:     577.05

LLM calls:  search_web("577.05 USD to EUR today")
Result:     "531.89 EUR"

LLM answers: "15% of 3847 is 577.05 USD = 531.89 EUR"
```

LLM never does math itself — it delegates to tools.

---

## 6. LangChain Agent — Full Code

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub

# ── 1. Define Tools ───────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input: math expression like '3847 * 0.15'"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 32°C, sunny"

tools = [calculator, get_weather]

# ── 2. LLM ───────────────────────────────────────────────
llm = ChatAnthropic(model="claude-opus-4-6", api_key="your-key")

# ── 3. Prompt (ReAct format) ──────────────────────────────
prompt = hub.pull("hwchase17/react")

# ── 4. Create Agent ───────────────────────────────────────
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ── 5. Run ────────────────────────────────────────────────
result = agent_executor.invoke({
    "input": "What is 15% of 3847? Also what is the weather in Chennai?"
})

print(result["output"])
```

Verbose output (shows thinking):
```
Thought: I need to calculate 15% of 3847
Action: calculator
Action Input: 3847 * 0.15
Observation: 577.05

Thought: Now get Chennai weather
Action: get_weather
Action Input: Chennai
Observation: Chennai: 32°C, sunny

Thought: I have all the answers
Final Answer: 15% of 3847 is 577.05. Chennai weather is 32°C, sunny.
```

---

## 7. Agent vs Chain

| | Chain | Agent |
|--|-------|-------|
| Steps | Fixed (you define order) | Dynamic (LLM decides) |
| Tools | No | Yes |
| Loop | No | Yes (until done) |
| Use when | Predictable flow | Open-ended tasks |

```
Chain:  input → step1 → step2 → step3 → output  (always same path)
Agent:  input → LLM decides what to do → loop → output (flexible path)
```

---

## 8. RAG + Agent (Powerful Combo)

Give the agent a retrieval tool:

```python
@tool
def search_docs(query: str) -> str:
    """Search internal documents for information."""
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])

tools = [search_docs, calculator, get_weather]
```

Now agent can:
1. Search private docs when needed
2. Do math when needed
3. Get live data when needed
4. Combine all results into one answer

---

## 9. What is LangChain?

**LangChain = framework that provides ready-made components to build LLM applications.**

```
Without LangChain:  500+ lines (write everything yourself)
With LangChain:     50 lines  (use built-in components)
```

Components:
```
Document Loaders → load PDF, web, CSV
Text Splitters   → chunk documents
Embeddings       → connect to embedding models
Vector Stores    → connect to ChromaDB, FAISS, Pinecone
LLMs             → connect to Claude, GPT, Gemini
Prompt Templates → reusable prompt structures
Chains (LCEL |)  → connect components with pipe operator
Agents           → LLM + tools + ReAct loop
Memory           → store conversation history
```

**Analogy:** LangChain is like Django for web development. Django gives you ready-made tools (auth, ORM, routing) so you build faster. LangChain gives ready-made LLM tools so you build AI apps faster.

---

## Summary

```
Agent     = LLM + Tools + ReAct loop
ReAct     = Thought → Action → Observation → repeat
Reasoning = thinking step before acting (prevents hallucination)
Tools     = functions LLM can call (search, calc, DB, API)
Chain     = fixed steps, no tools
Agent     = dynamic steps, uses tools as needed
LangChain = framework with all components ready-made
```
