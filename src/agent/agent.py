from __future__ import annotations

"""
Main Crowd Analysis AI Agent.

This file turns the dashboard assistant into a data-grounded AI agent.

How it works:
1. User asks a free question.
2. Python tools retrieve relevant facts from CSV files.
3. If an API key exists, Gemini/OpenAI explains those facts naturally.
4. If no key or the API fails, the agent falls back to reliable rule-based answers.

Recommended Gemini package:
    pip install -U google-genai

.env example:
    GOOGLE_API_KEY=your_key_here
    CROWD_AGENT_PROVIDER=gemini
    CROWD_AGENT_DEBUG=1
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os
import traceback

from src.agent.prompts import SYSTEM_PROMPT, build_messages, build_user_prompt
from src.agent.tools import (
    LoadedCrowdData,
    answer_rule_based,
    build_context_for_question,
    build_context_text_for_question,
    extract_zone_from_question,
    load_crowd_data,
    resolve_zone_name,
)


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class AgentResponse:
    answer: str
    mode: str
    used_llm: bool
    selected_zone: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    context_text: Optional[str] = None
    error: Optional[str] = None


def load_dotenv_if_available(project_root: Path) -> None:
    """
    Lightweight .env loader.

    Supports:
        GOOGLE_API_KEY=...
        OPENAI_API_KEY=...
        CROWD_AGENT_PROVIDER=gemini/openai/rule/auto
        CROWD_AGENT_DEBUG=1
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


class CrowdAnalysisAgent:
    """
    Data-grounded AI assistant for the crowd monitoring dashboard.

    Usage:
        agent = CrowdAnalysisAgent(project_root=PROJECT_ROOT)
        answer = agent.answer("At 1:00 what was the risk classification in each zone?")
    """

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        provider: Optional[str] = None,
        use_llm: Optional[bool] = None,
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        openai_model: str = DEFAULT_OPENAI_MODEL,
        debug: Optional[bool] = None,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
        load_dotenv_if_available(self.project_root)

        self.provider = (provider or os.getenv("CROWD_AGENT_PROVIDER", "auto")).lower().strip()
        self.gemini_model = gemini_model
        self.openai_model = openai_model

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
        if self._data is None:
            self._data = load_crowd_data(self.project_root)
        return self._data

    def reload_data(self) -> None:
        self._data = load_crowd_data(self.project_root)

    # ========================================================
    # MAIN ANSWER METHODS
    # ========================================================

    def answer(
        self,
        question: str,
        selected_zone: Optional[str] = None,
        return_details: bool = False,
    ) -> str | AgentResponse:
        question = str(question or "").strip()

        if not question:
            response = AgentResponse(
                answer="Please ask a question about the crowd analysis results.",
                mode="empty",
                used_llm=False,
                selected_zone=selected_zone,
            )
            return response if return_details else response.answer

        try:
            data = self.data

            resolved_selected_zone = None
            if selected_zone:
                resolved_selected_zone = resolve_zone_name(data, selected_zone) or selected_zone

            zone_from_question = extract_zone_from_question(data, question)
            zone_for_context = zone_from_question or resolved_selected_zone

            context_dict = build_context_for_question(
                data=data,
                question=question,
                selected_zone=zone_for_context,
            )

            context_text = build_context_text_for_question(
                data=data,
                question=question,
                selected_zone=zone_for_context,
            )

            if self.use_llm:
                llm_answer = self._try_llm_answer(
                    question=question,
                    context_text=context_text,
                    selected_zone=zone_for_context,
                )

                if llm_answer:
                    response = AgentResponse(
                        answer=llm_answer,
                        mode=self._active_provider_name(),
                        used_llm=True,
                        selected_zone=zone_for_context,
                        context=context_dict if self.debug else None,
                        context_text=context_text if self.debug else None,
                    )
                    return response if return_details else response.answer

            fallback = answer_rule_based(
                data=data,
                question=question,
                selected_zone=zone_for_context,
            )

            response = AgentResponse(
                answer=fallback,
                mode="rule",
                used_llm=False,
                selected_zone=zone_for_context,
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
                error=error_text,
            )

            return response if return_details else response.answer

    def quick_answer(self, action: str, selected_zone: Optional[str] = None) -> str:
        action = str(action or "").strip().lower()

        if "risky" in action or "risk" in action:
            return str(
                self.answer(
                    "Which zone is the most risky and why?",
                    selected_zone=selected_zone,
                )
            )

        if "peak" in action:
            return str(
                self.answer(
                    "When was the peak crowd moment and what happened around it?",
                    selected_zone=selected_zone,
                )
            )

        if "selected" in action or "explain" in action or "zone" in action:
            zone = selected_zone or "the selected zone"
            return str(
                self.answer(
                    f"Explain {zone} using the analysis results.",
                    selected_zone=selected_zone,
                )
            )

        if "anomaly" in action or "alert" in action:
            return str(
                self.answer(
                    "Summarize the anomaly detection and spike events.",
                    selected_zone=selected_zone,
                )
            )

        return str(self.answer(action, selected_zone=selected_zone))

    # ========================================================
    # PROVIDER SELECTION
    # ========================================================

    def _active_provider_name(self) -> str:
        if self.provider in {"gemini", "google"}:
            return "gemini"

        if self.provider == "openai":
            return "openai"

        if self.provider == "rule":
            return "rule"

        if os.getenv("GOOGLE_API_KEY"):
            return "gemini"

        if os.getenv("OPENAI_API_KEY"):
            return "openai"

        return "rule"

    def _try_llm_answer(
        self,
        question: str,
        context_text: str,
        selected_zone: Optional[str] = None,
    ) -> Optional[str]:
        provider = self._active_provider_name()

        if provider == "gemini":
            return self._try_gemini_answer(
                question=question,
                context_text=context_text,
                selected_zone=selected_zone,
            )

        if provider == "openai":
            return self._try_openai_answer(
                question=question,
                context_text=context_text,
                selected_zone=selected_zone,
            )

        return None

    # ========================================================
    # GEMINI — NEW SDK: google-genai
    # ========================================================

    def _try_gemini_answer(
        self,
        question: str,
        context_text: str,
        selected_zone: Optional[str] = None,
    ) -> Optional[str]:
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            return None

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            prompt = build_user_prompt(
                question=question,
                context=context_text,
                selected_zone=selected_zone,
            )

            response = client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.25,
                    top_p=0.85,
                    max_output_tokens=900,
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
        selected_zone: Optional[str] = None,
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
            )

            response = client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.25,
                max_tokens=900,
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


if __name__ == "__main__":
    agent = CrowdAnalysisAgent()

    tests = [
        "What are you?",
        "At 1:00 what was the risk classification in each zone?",
        "Which zone is most risky?",
        "When was the peak moment?",
        "Explain sidewalk_right.",
        "Compare sidewalk_right and crosswalk_main.",
    ]

    for q in tests:
        print("=" * 90)
        print("Q:", q)
        print("-" * 90)
        result = agent.answer(q, selected_zone="sidewalk_right", return_details=True)
        print("MODE:", result.mode)
        print("USED_LLM:", result.used_llm)
        print(result.answer)
        print()