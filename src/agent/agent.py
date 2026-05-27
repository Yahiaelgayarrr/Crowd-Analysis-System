from __future__ import annotations

"""
Main Crowd Analysis AI Agent.

This file connects:
- src/agent/tools.py   -> reads CSVs, detects intent, retrieves exact facts
- src/agent/prompts.py -> builds thesis-safe, intent-aware LLM prompts
- src/dashboard/app.py -> calls CrowdAnalysisAgent.answer(...)

Agent flow:
1. User asks a free question.
2. tools.detect_intent() identifies the request type, zones, timestamp, chart, etc.
3. tools.build_context_text_for_question() extracts compact factual context from CSV outputs.
4. Gemini/OpenAI explains the facts naturally.
5. If API fails, answer_rule_based() gives a deterministic data-grounded fallback.

Recommended Gemini package:
    pip install -U google-genai

.env example:
    GOOGLE_API_KEY=your_key_here
    CROWD_AGENT_PROVIDER=gemini
    CROWD_AGENT_DEBUG=1

Important:
- Do NOT commit .env to GitHub.
- The LLM never receives full CSV files directly.
- Python tools extract the relevant facts first.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os
import traceback

from src.agent.prompts import (
    SYSTEM_PROMPT,
    build_messages,
    build_system_prompt_for_intent,
    build_user_prompt,
)
from src.agent.tools import (
    LoadedCrowdData,
    answer_rule_based,
    build_context_for_question,
    build_context_text_for_question,
    detect_intent,
    extract_zone_from_question,
    load_crowd_data,
    resolve_zone_name,
)


# ============================================================
# DEFAULT MODELS
# ============================================================

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


# ============================================================
# RESPONSE OBJECT
# ============================================================

@dataclass
class AgentResponse:
    answer: str
    mode: str
    used_llm: bool
    selected_zone: Optional[str] = None
    intent: Optional[str] = None
    chart_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    context_text: Optional[str] = None
    error: Optional[str] = None


# ============================================================
# ENV LOADING
# ============================================================

def load_dotenv_if_available(project_root: Path) -> None:
    """
    Lightweight .env loader.

    Supports:
        GOOGLE_API_KEY=...
        OPENAI_API_KEY=...
        CROWD_AGENT_PROVIDER=gemini/openai/rule/auto
        CROWD_AGENT_DEBUG=1
        CROWD_AGENT_GEMINI_MODEL=gemini-2.5-flash-lite
        CROWD_AGENT_OPENAI_MODEL=gpt-4o-mini
    """
    env_path = project_root / ".env"

    if not env_path.exists():
        return

    try:
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value

    except Exception:
        return


# ============================================================
# AGENT
# ============================================================

class CrowdAnalysisAgent:
    """
    Data-grounded AI assistant for the crowd monitoring dashboard.

    Usage:
        agent = CrowdAnalysisAgent(project_root=PROJECT_ROOT)
        answer = agent.answer(
            "At 1:00 what was the risk classification in each zone?",
            selected_zone="sidewalk_right",
        )
    """

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        provider: Optional[str] = None,
        use_llm: Optional[bool] = None,
        gemini_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )

        load_dotenv_if_available(self.project_root)

        self.provider = (provider or os.getenv("CROWD_AGENT_PROVIDER", "auto")).lower().strip()

        self.gemini_model = (
            gemini_model
            or os.getenv("CROWD_AGENT_GEMINI_MODEL")
            or DEFAULT_GEMINI_MODEL
        )

        self.openai_model = (
            openai_model
            or os.getenv("CROWD_AGENT_OPENAI_MODEL")
            or DEFAULT_OPENAI_MODEL
        )

        if use_llm is None:
            self.use_llm = self.provider != "rule"
        else:
            self.use_llm = bool(use_llm)

        if debug is None:
            self.debug = os.getenv("CROWD_AGENT_DEBUG", "0").strip() == "1"
        else:
            self.debug = bool(debug)

        self._data: Optional[LoadedCrowdData] = None

    # ========================================================
    # DATA
    # ========================================================

    @property
    def data(self) -> LoadedCrowdData:
        """
        Lazy-load data once per agent instance.
        """
        if self._data is None:
            self._data = load_crowd_data(self.project_root)
        return self._data

    def reload_data(self) -> None:
        """
        Force reload if CSV outputs were replaced.
        """
        self._data = load_crowd_data(self.project_root)

    # ========================================================
    # MAIN ANSWER METHOD
    # ========================================================

    def answer(
        self,
        question: str,
        selected_zone: Optional[str] = None,
        return_details: bool = False,
    ) -> str | AgentResponse:
        """
        Main method used by the dashboard.

        Parameters:
            question:
                User's free-text question.

            selected_zone:
                Zone currently selected in dashboard. This is used only when useful.
                The tools router ignores it for all-zone questions such as:
                "At 1:00 list every zone."

            return_details:
                If True, returns AgentResponse with mode/intent/debug info.
                If False, returns answer text only.
        """
        question = str(question or "").strip()

        if not question:
            response = AgentResponse(
                answer="Please ask a question about the crowd analysis results.",
                mode="empty",
                used_llm=False,
                selected_zone=selected_zone,
                intent="empty",
            )
            return response if return_details else response.answer

        try:
            data = self.data

            resolved_selected_zone = None
            if selected_zone:
                resolved_selected_zone = resolve_zone_name(data, selected_zone) or selected_zone

            # Detect intent once here so both LLM and debug details know the same route.
            intent_info = detect_intent(
                data=data,
                question=question,
                selected_zone=resolved_selected_zone,
            )

            intent = str(intent_info.get("intent", "general"))
            chart_name = intent_info.get("chart_name")
            zone_for_context = intent_info.get("zone_to_use")

            # If tools did not decide zone_to_use, allow explicitly mentioned zone.
            if zone_for_context is None:
                zone_from_question = extract_zone_from_question(data, question)
                zone_for_context = zone_from_question

            context_dict = build_context_for_question(
                data=data,
                question=question,
                selected_zone=resolved_selected_zone,
            )

            context_text = build_context_text_for_question(
                data=data,
                question=question,
                selected_zone=resolved_selected_zone,
            )

            if self.use_llm:
                llm_answer = self._try_llm_answer(
                    question=question,
                    context_text=context_text,
                    selected_zone=zone_for_context or resolved_selected_zone,
                    intent=intent,
                    chart_name=chart_name,
                )

                if llm_answer:
                    response = AgentResponse(
                        answer=llm_answer,
                        mode=self._active_provider_name(),
                        used_llm=True,
                        selected_zone=zone_for_context or resolved_selected_zone,
                        intent=intent,
                        chart_name=chart_name,
                        context=context_dict if self.debug else None,
                        context_text=context_text if self.debug else None,
                    )
                    return response if return_details else response.answer

            fallback_answer = answer_rule_based(
                data=data,
                question=question,
                selected_zone=resolved_selected_zone,
            )

            response = AgentResponse(
                answer=fallback_answer,
                mode="rule",
                used_llm=False,
                selected_zone=zone_for_context or resolved_selected_zone,
                intent=intent,
                chart_name=chart_name,
                context=context_dict if self.debug else None,
                context_text=context_text if self.debug else None,
            )

            return response if return_details else response.answer

        except Exception as exc:
            error_text = str(exc)

            if self.debug:
                error_text += "\n\n" + traceback.format_exc()

            response = AgentResponse(
                answer=(
                    "I could not answer because the analysis data could not be loaded or processed. "
                    f"Error: {error_text}"
                ),
                mode="error",
                used_llm=False,
                selected_zone=selected_zone,
                intent="error",
                error=error_text,
            )

            return response if return_details else response.answer

    # ========================================================
    # QUICK BUTTON SUPPORT
    # ========================================================

    def quick_answer(self, action: str, selected_zone: Optional[str] = None) -> str:
        """
        Helper for dashboard quick buttons.
        """
        action_lower = str(action or "").strip().lower()

        if "risky" in action_lower or "risk" in action_lower:
            return str(
                self.answer(
                    "Which zone is the most risky and why? Give evidence from the data.",
                    selected_zone=selected_zone,
                )
            )

        if "peak" in action_lower:
            return str(
                self.answer(
                    "When was the peak crowd moment and what happened around it?",
                    selected_zone=selected_zone,
                )
            )

        if "selected" in action_lower or "explain" in action_lower or "zone" in action_lower:
            zone = selected_zone or "the selected zone"
            return str(
                self.answer(
                    f"Explain {zone} using the analysis results.",
                    selected_zone=selected_zone,
                )
            )

        if "anomaly" in action_lower or "alert" in action_lower or "spike" in action_lower:
            return str(
                self.answer(
                    "Summarize the anomaly detection and spike events. Explain what they mean.",
                    selected_zone=selected_zone,
                )
            )

        if "temporal" in action_lower:
            return str(
                self.answer(
                    "Explain the temporal analysis and give a thesis-safe interpretation.",
                    selected_zone=selected_zone,
                )
            )

        if "recommend" in action_lower or "decision" in action_lower:
            return str(
                self.answer(
                    "Give evidence-backed recommendations for crowd monitoring.",
                    selected_zone=selected_zone,
                )
            )

        return str(self.answer(action, selected_zone=selected_zone))

    # ========================================================
    # PROVIDER SELECTION
    # ========================================================

    def _active_provider_name(self) -> str:
        """
        Decide which backend is active.
        """
        if self.provider in {"gemini", "google"}:
            return "gemini"

        if self.provider == "openai":
            return "openai"

        if self.provider == "rule":
            return "rule"

        # auto mode
        if os.getenv("GOOGLE_API_KEY"):
            return "gemini"

        if os.getenv("OPENAI_API_KEY"):
            return "openai"

        return "rule"

    def _try_llm_answer(
        self,
        question: str,
        context_text: str,
        selected_zone: Optional[str],
        intent: str,
        chart_name: Optional[str],
    ) -> Optional[str]:
        provider = self._active_provider_name()

        if provider == "gemini":
            return self._try_gemini_answer(
                question=question,
                context_text=context_text,
                selected_zone=selected_zone,
                intent=intent,
                chart_name=chart_name,
            )

        if provider == "openai":
            return self._try_openai_answer(
                question=question,
                context_text=context_text,
                selected_zone=selected_zone,
                intent=intent,
                chart_name=chart_name,
            )

        return None

    # ========================================================
    # GEMINI — google-genai
    # ========================================================

    def _try_gemini_answer(
        self,
        question: str,
        context_text: str,
        selected_zone: Optional[str],
        intent: str,
        chart_name: Optional[str],
    ) -> Optional[str]:
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            return None

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            system_instruction = build_system_prompt_for_intent(
                intent=intent,
                chart_name=chart_name,
            )

            prompt = build_user_prompt(
                question=question,
                context=context_text,
                selected_zone=selected_zone,
            )

            response = client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.25,
                    top_p=0.85,
                    max_output_tokens=1000,
                ),
            )

            text = getattr(response, "text", None)

            if text:
                return self._clean_answer(text)

            return None

        except Exception:
            if self.debug:
                print("Gemini call failed:")
                traceback.print_exc()
            return None

    # ========================================================
    # OPENAI
    # ========================================================

    def _try_openai_answer(
        self,
        question: str,
        context_text: str,
        selected_zone: Optional[str],
        intent: str,
        chart_name: Optional[str],
    ) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return None

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            messages = build_messages(
                question=question,
                context=context_text,
                selected_zone=selected_zone,
                intent=intent,
                chart_name=chart_name,
            )

            response = client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.25,
                max_tokens=1000,
            )

            text = response.choices[0].message.content

            if text:
                return self._clean_answer(text)

            return None

        except Exception:
            if self.debug:
                print("OpenAI call failed:")
                traceback.print_exc()
            return None

    # ========================================================
    # CLEANUP
    # ========================================================

    @staticmethod
    def _clean_answer(text: str) -> str:
        text = str(text or "").strip()

        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text


# ============================================================
# FACTORY
# ============================================================

def get_agent(
    project_root: Optional[str | Path] = None,
    provider: Optional[str] = None,
    use_llm: Optional[bool] = None,
) -> CrowdAnalysisAgent:
    return CrowdAnalysisAgent(
        project_root=project_root,
        provider=provider,
        use_llm=use_llm,
    )


# ============================================================
# CLI TEST
# ============================================================

def _demo() -> None:
    agent = CrowdAnalysisAgent()

    tests = [
        "What can you do?",
        "At 1:00 what was the risk classification in each zone?",
        "Explain temporal analysis and give a recommendation.",
        "Explain the correlation heatmap.",
        "Give evidence-backed recommendations for crowd monitoring.",
        "Compare sidewalk_right and crosswalk_main.",
    ]

    for question in tests:
        print("=" * 100)
        print("Q:", question)
        print("-" * 100)

        result = agent.answer(
            question,
            selected_zone="sidewalk_right",
            return_details=True,
        )

        print("MODE:", result.mode)
        print("USED_LLM:", result.used_llm)
        print("INTENT:", result.intent)
        print("CHART:", result.chart_name)
        print()
        print(result.answer)
        print()


if __name__ == "__main__":
    _demo()