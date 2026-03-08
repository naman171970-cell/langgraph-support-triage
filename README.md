# langgraph-support-triage
Stateful multi-agent support orchestration engine with deterministic guardrails

# Enterprise Support Triage: Multi-Agent Orchestration

This repository contains a prototype for a stateful, multi-agent support triage system built with LangGraph and Claude 4.6 Sonnet.

## The Business Problem
Automating customer support with LLMs introduces severe compliance and financial risks (e.g., unauthorized refunds). This architecture solves the trust and safety bottleneck.

## The Architecture
1. **Agentic Orchestration:** A Supervisor node dynamically routes tickets to specialized domain agents (Billing, Tech Support).
2. **Stateful Memory:** Nodes read and write to a shared `AgentState` payload to prevent context loss.
3. **Deterministic Guardrails:** A pure Python permission gate intercepts high-risk LLM output (e.g., "refund", "override") and forces a human-in-the-loop escalation.

## Files
* `support_triage.py`: Contains the state definition, agent prompts, and routing logic.
* `langgraph.json`: Configuration for the LangGraph CLI execution environment.
