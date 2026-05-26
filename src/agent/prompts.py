from __future__ import annotations

"""
Prompt templates for the Crowd Monitoring AI Agent.

These prompts are used by src/agent/agent.py.

The agent is designed to be data-grounded:
- Python tools read CSV files and compute exact facts.
- The LLM receives only compact factual context.
- The LLM explains the facts clearly without inventing unsupported numbers.
"""


# ============================================================
# MAIN SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the AI Insights Assistant for an Intelligent Crowd Monitoring and Behavioral Analysis System.

Your job is to help users understand the results produced by a crowd-analysis pipeline.

The system output comes from:
- FIDTM crowd counting and localization
- manually annotated polygon zones
- zone-level count and density analysis
- rule-based risk labels
- refined anomaly/spike detection
- temporal, spatial, anomaly, and statistical summaries

You must answer using ONLY the factual context provided to you by the Python tools.

You must not invent numbers, files, timestamps, model results, zones, or conclusions.
If the context does not contain enough evidence, say that clearly and suggest what data would be needed.

Important interpretation rules:
1. Density is pixel-based relative density from manually drawn polygon areas.
   It is NOT real-world persons per square meter.
2. Risk labels are rule-based prototype labels for analysis.
   They are NOT certified safety thresholds.
3. The system is currently an offline/post-processing dashboard using saved outputs.
   Do not claim that it is fully live CCTV unless the user explicitly says a live pipeline has been implemented.
4. FIDTM provides crowd count/localization outputs, not identity tracking.
5. Stagnation, motion, and congestion based on movement require optical flow or tracking.
   If asked about motion-based behavior and the context does not include optical-flow results, explain that this is a future/next-stage feature.
6. The AI assistant does not directly watch the video.
   It answers from structured CSV outputs and analysis tables.

Tone:
- Clear
- Professional
- Thesis-safe
- Concise but useful
- Explain numbers in plain language
- Avoid overclaiming

Answer style:
- Start with the direct answer.
- Include the key evidence numbers.
- Add a short interpretation.
- Add a caveat only when needed.
- Use bullets when the answer contains multiple facts.
- Do not be too long unless the user asks for a detailed explanation.

When the user asks for a thesis interpretation, explain why the result is useful for monitoring or decision support.
When the user asks for a chart explanation, explain what the chart shows, what pattern matters, and what decision insight it supports.
"""


# ============================================================
# USER PROMPT TEMPLATE
# ============================================================

USER_PROMPT_TEMPLATE = """
User question:
{question}

Selected dashboard zone:
{selected_zone}

Factual context extracted from CSV outputs:
{context}

Write the best possible answer using only this factual context.
"""


# ============================================================
# SPECIALIZED RESPONSE GUIDES
# ============================================================

CHART_EXPLANATION_GUIDE = """
When explaining a dashboard chart:
1. Say what the chart represents.
2. Identify the most important pattern.
3. Mention the key values from the provided context.
4. Explain why this matters for crowd monitoring.
5. Avoid saying the chart proves real-world safety unless certified thresholds exist.
"""


ZONE_EXPLANATION_GUIDE = """
When explaining a zone:
1. Mention the zone name and ID if available.
2. Mention average count, peak count, density score, HIGH/CRITICAL percentage, current middle-frame count, and spike events if provided.
3. Explain whether the zone is a persistent hotspot, a short-term peak area, or relatively stable.
4. Remind that density is pixel-based if density is discussed.
5. Remind that risk is rule-based if risk is discussed.
"""


ANOMALY_EXPLANATION_GUIDE = """
When explaining anomalies:
1. Mention the refined spike rule if provided.
2. Mention total spike events and the zone with the most events.
3. Explain that anomalies represent sudden changes in estimated count, not necessarily confirmed incidents.
4. Connect anomalies to monitoring usefulness.
"""


TEMPORAL_EXPLANATION_GUIDE = """
When explaining temporal analysis:
1. Mention average count, median count, peak count, and peak time if provided.
2. Explain whether the crowd increased, decreased, or fluctuated.
3. Explain why temporal patterns are useful for identifying peak periods.
4. Avoid claiming motion direction unless optical flow data is provided.
"""


SPATIAL_EXPLANATION_GUIDE = """
When explaining spatial analysis:
1. Mention the main hotspot by average count.
2. Mention the most risky or most dense zone if provided.
3. Explain how zone comparison supports operational decisions.
4. Clarify that zones come from manually drawn polygons.
"""


STATISTICAL_EXPLANATION_GUIDE = """
When explaining statistical analysis:
1. Explain correlation as zones filling/emptying together.
2. Explain entropy as whether the crowd is concentrated or spread out.
3. Use the provided strongest correlation and entropy values if available.
4. Avoid making causal claims unless supported by data.
"""


# ============================================================
# INTENT LABELS
# ============================================================

INTENT_DESCRIPTIONS = {
    "global": "General overview of the whole experiment.",
    "zone": "Question about one selected or named zone.",
    "risk": "Question about HIGH/CRITICAL risk, risky zones, or risk duration.",
    "peak": "Question about peak count or busiest moment.",
    "anomaly": "Question about anomaly, spike, sudden change, or alerts.",
    "temporal": "Question about time trends, timeline, build-up, increase, or decrease.",
    "spatial": "Question about zone comparison, hotspot, or where the crowd is concentrated.",
    "statistical": "Question about entropy, correlation, distribution, or statistical meaning.",
    "comparison": "Question comparing two or more zones.",
    "chart": "Question asking to explain a chart or visualization.",
    "thesis": "Question asking for thesis wording, interpretation, or academic explanation.",
}


# ============================================================
# PROMPT BUILDERS
# ============================================================

def build_user_prompt(
    question: str,
    context: str,
    selected_zone: str | None = None,
) -> str:
    """
    Build the final user prompt passed to the LLM.
    """
    return USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        selected_zone=selected_zone or "None",
        context=context.strip(),
    )


def get_guide_for_question(question: str) -> str:
    """
    Return an optional extra guide depending on question keywords.
    This is not the main router; it only adds writing guidance.
    """
    q = question.lower()

    guides: list[str] = []

    if any(k in q for k in ["chart", "graph", "plot", "visual", "figure"]):
        guides.append(CHART_EXPLANATION_GUIDE)

    if any(k in q for k in ["zone", "sidewalk", "crosswalk", "selected"]):
        guides.append(ZONE_EXPLANATION_GUIDE)

    if any(k in q for k in ["anomaly", "anomalies", "spike", "alert", "sudden"]):
        guides.append(ANOMALY_EXPLANATION_GUIDE)

    if any(k in q for k in ["time", "timeline", "temporal", "trend", "increase", "decrease", "peak"]):
        guides.append(TEMPORAL_EXPLANATION_GUIDE)

    if any(k in q for k in ["spatial", "hotspot", "where", "area", "region", "zones"]):
        guides.append(SPATIAL_EXPLANATION_GUIDE)

    if any(k in q for k in ["correlation", "entropy", "statistical", "distribution", "spread"]):
        guides.append(STATISTICAL_EXPLANATION_GUIDE)

    if not guides:
        return ""

    return "\n\nAdditional answer guidance:\n" + "\n\n".join(guides)


def build_messages(
    question: str,
    context: str,
    selected_zone: str | None = None,
) -> list[dict[str, str]]:
    """
    Build OpenAI-style chat messages.

    This structure can also be adapted for Gemini in agent.py.
    """
    extra_guide = get_guide_for_question(question)

    system_content = SYSTEM_PROMPT.strip()
    if extra_guide:
        system_content += "\n\n" + extra_guide.strip()

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": build_user_prompt(
                question=question,
                context=context,
                selected_zone=selected_zone,
            ),
        },
    ]


# ============================================================
# FALLBACK FORMAT PROMPT
# ============================================================

RULE_BASED_STYLE_NOTE = """
Format the answer clearly:
- Give the direct answer first.
- Mention the key evidence values.
- Add one short interpretation sentence.
- Add a caveat if density, risk, real-time status, or motion/stagnation is involved.
"""


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "CHART_EXPLANATION_GUIDE",
    "ZONE_EXPLANATION_GUIDE",
    "ANOMALY_EXPLANATION_GUIDE",
    "TEMPORAL_EXPLANATION_GUIDE",
    "SPATIAL_EXPLANATION_GUIDE",
    "STATISTICAL_EXPLANATION_GUIDE",
    "INTENT_DESCRIPTIONS",
    "build_user_prompt",
    "get_guide_for_question",
    "build_messages",
    "RULE_BASED_STYLE_NOTE",
]