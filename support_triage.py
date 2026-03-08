"""
Merchant Support Triage System - LangGraph Multi-Agent Application

This script implements a support ticket triage workflow using LangGraph.
Designed for use with LangGraph Studio - the compiled graph is assigned to `graph`.
"""

import base64
import os
import re
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.content import create_image_block
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict


# =============================================================================
# 1. State Definition
# =============================================================================


class AgentState(TypedDict):
    """Shared state passed between all nodes in the triage workflow."""

    ticket_text: str
    image_path: str  # Optional path to a local screenshot
    extracted_problem: str
    assigned_agent: str
    proposed_action: str
    permission_granted: bool
    escalation_report: NotRequired[str]  # Set when routed to human review


# =============================================================================
# 2. LLM Setup
# =============================================================================

llm = ChatAnthropic(model="claude-sonnet-4-6")


# =============================================================================
# 3. Node Functions
# =============================================================================


def triage_node(state: AgentState) -> dict:
    """
    Extracts and summarizes the core issue from the ticket.
    If image_path is provided, loads the image, base64-encodes it, and sends
    both ticket text and image to the LLM for multimodal analysis.
    """
    ticket_text = state.get("ticket_text", "")
    image_path = state.get("image_path", "")

    # Build the message content: text + optional image
    content = []

    if image_path and image_path.strip():
        # Open the local image file, encode to base64, and add to message
        try:
            with open(image_path, "rb") as image_file:
                base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            # Infer MIME type from file extension for Anthropic compatibility
            ext = os.path.splitext(image_path)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_map.get(ext, "image/png")
            content.append(create_image_block(base64=base64_data, mime_type=mime_type))
        except (FileNotFoundError, OSError):
            # If image cannot be loaded, proceed with text only
            pass

    # Add ticket text (as first element for context, or only element if no image)
    content.append({"type": "text", "text": ticket_text})

    # Construct multimodal HumanMessage for Anthropic
    message = HumanMessage(content=content)

    prompt = (
        "You are a support triage analyst. Analyze the following support ticket "
        "(and any attached screenshot, if present) and extract the core problem in 1-3 concise sentences. "
        "Focus on the essential issue the customer is facing. Output only the problem summary, nothing else."
    )

    response = llm.invoke([SystemMessage(content=prompt), message])
    extracted_problem = response.content if isinstance(response.content, str) else str(response.content)

    return {"extracted_problem": extracted_problem.strip()}


def supervisor_node(state: AgentState) -> dict:
    """
    Routes the ticket to the appropriate specialist agent.
    Outputs exactly one of: Security_Agent, Billing_Agent, or Product_Expert_Agent.
    """
    extracted_problem = state.get("extracted_problem", "")

    prompt = f"""You are a support supervisor. Based on the following extracted problem, assign it to exactly ONE agent.

Extracted problem: {extracted_problem}

You must output ONLY one of these three strings, with no other text:
- Security_Agent (for security, access, authentication, or account lockout issues)
- Billing_Agent (for payments, refunds, subscriptions, or billing questions)
- Product_Expert_Agent (for product usage, dashboard navigation, features, or how-to questions)

Output only the agent name:"""

    response = llm.invoke(prompt)
    raw_output = response.content if isinstance(response.content, str) else str(response.content)

    # Parse to ensure we get exactly one valid agent name (in case LLM adds extra text)
    valid_agents = ("Security_Agent", "Billing_Agent", "Product_Expert_Agent")
    match = re.search(r"(" + "|".join(re.escape(a) for a in valid_agents) + ")", raw_output, re.I)
    assigned_agent = match.group(1) if match else "Product_Expert_Agent"

    return {"assigned_agent": assigned_agent}


def specialist_node(state: AgentState) -> dict:
    """
    Drafts a solution based on the assigned agent role.
    Product_Expert_Agent must provide step-by-step dashboard navigation instructions.
    """
    extracted_problem = state.get("extracted_problem", "")
    assigned_agent = state.get("assigned_agent", "Product_Expert_Agent")

    role_instructions = {
        "Security_Agent": (
            "You are the Security_Agent. Provide clear security-related guidance: "
            "account recovery, access issues, authentication, or security best practices."
        ),
        "Billing_Agent": (
            "You are the Billing_Agent. Provide billing and payment guidance: "
            "refunds, subscriptions, invoices, or payment issues."
        ),
        "Product_Expert_Agent": (
            "You are the Product_Expert_Agent. You MUST provide step-by-step dashboard "
            "navigation instructions. Format your response with numbered steps (e.g., 1. Go to..., 2. Click..., etc.)."
        ),
    }

    instruction = role_instructions.get(
        assigned_agent,
        "You are a support specialist. Provide helpful, actionable guidance.",
    )

    prompt = f"""You are acting as the {assigned_agent}.

{instruction}

Extracted problem: {extracted_problem}

Draft a proposed action/solution for the customer. Be specific and actionable. Output only the proposed action, no preamble."""

    response = llm.invoke(prompt)
    proposed_action = response.content if isinstance(response.content, str) else str(response.content)

    return {"proposed_action": proposed_action.strip()}


def permission_node(state: AgentState) -> dict:
    """
    Pure Python function (no LLM). Checks if the proposed action requires
    human approval. Sensitive actions (override, refund, revoke, lockout) are denied.
    """
    proposed_action = state.get("proposed_action", "")
    sensitive_keywords = ("override", "refund", "revoke", "lockout")
    proposed_lower = proposed_action.lower()
    # If any sensitive keyword is present, permission is denied
    permission_granted = not any(kw in proposed_lower for kw in sensitive_keywords)
    return {"permission_granted": permission_granted}


def human_escalation_node(state: AgentState) -> dict:
    """
    Formats the ticket for human review when the proposed action requires approval.
    Combines problem, assigned agent, and proposed action into a clear escalation report.
    """
    extracted_problem = state.get("extracted_problem", "")
    assigned_agent = state.get("assigned_agent", "")
    proposed_action = state.get("proposed_action", "")

    escalation_report = (
        "=== HUMAN REVIEW REQUIRED ===\n\n"
        f"Problem: {extracted_problem}\n\n"
        f"Assigned Agent: {assigned_agent}\n\n"
        f"Proposed Action: {proposed_action}\n\n"
        "This action requires manual approval before execution."
    )

    return {"escalation_report": escalation_report}


# =============================================================================
# 4. Conditional Routing
# =============================================================================


def route_after_permission(state: AgentState) -> Literal["end", "escalate"]:
    """
    Routes to END if permission is granted, otherwise to human_escalation_node.
    """
    if state.get("permission_granted", True):
        return "end"
    return "escalate"


# =============================================================================
# 5. Graph Construction
# =============================================================================

# Create the state graph with AgentState schema
workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("triage_node", triage_node)
workflow.add_node("supervisor_node", supervisor_node)
workflow.add_node("specialist_node", specialist_node)
workflow.add_node("permission_node", permission_node)
workflow.add_node("human_escalation_node", human_escalation_node)

# Linear flow: START -> triage -> supervisor -> specialist -> permission
workflow.add_edge(START, "triage_node")
workflow.add_edge("triage_node", "supervisor_node")
workflow.add_edge("supervisor_node", "specialist_node")
workflow.add_edge("specialist_node", "permission_node")

# Conditional edge: permission_granted True -> END, False -> human_escalation_node
workflow.add_conditional_edges(
    "permission_node",
    route_after_permission,
    {
        "end": END,
        "escalate": "human_escalation_node",
    },
)

# Human escalation node always leads to END
workflow.add_edge("human_escalation_node", END)

# Compile the graph for LangGraph Studio (must be assigned to variable named 'graph')
graph = workflow.compile()
