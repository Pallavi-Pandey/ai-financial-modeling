"""
AI-powered insights engine.

Provider fallback chain (first success wins):
  1. Gemini key 1  (GEMINI_API_KEY_1)
  2. Gemini key 2  (GEMINI_API_KEY_2)
  3. Pollinations.ai  (no key required, openai model)
  4. OllamaFreeAPI    (no key required, llama3.2)
  5. Claude           (ANTHROPIC_API_KEY)
  6. Graceful error message
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ANTHROPIC_API_KEY, GEMINI_API_KEYS, GEMINI_MODEL

_POLLINATIONS_URL = "https://text.pollinations.ai/"
_POLLINATIONS_MODEL = "openai"

_OLLAMA_MODEL = "llama3.2:3b"


class AIInsightsEngine:
    """Generate LLM-powered financial insights.

    Tries providers in order: Gemini 1 → Gemini 2 → Pollinations → OllamaFree → Claude.
    """

    def __init__(self) -> None:
        self._gemini_keys: List[str] = [k for k in GEMINI_API_KEYS if k]
        self._anthropic_key: str = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY", "")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_performance_insights(
        self,
        metrics_dict: Dict[str, Any],
        feature_importance_df: pd.DataFrame,
    ) -> str:
        top_features = (
            feature_importance_df.head(5).to_string(index=False)
            if not feature_importance_df.empty
            else "No feature importance data available."
        )

        system = (
            "You are an expert financial analyst and business strategist. "
            "Provide clear, actionable, and data-driven insights. Be concise and focus on business value."
        )
        prompt = f"""You are a financial data scientist reviewing ML model performance
for a business forecasting system.

## Model Evaluation Metrics
- MAE  (Mean Absolute Error):  ${metrics_dict.get('MAE', 'N/A'):,.2f}
- RMSE (Root Mean Sq. Error):  ${metrics_dict.get('RMSE', 'N/A'):,.2f}
- R²   (Coefficient of Det.):  {metrics_dict.get('R2', 'N/A')}
- MAPE (Mean Abs. % Error):    {metrics_dict.get('MAPE', 'N/A')}%

## Top 5 Feature Importances
{top_features}

Provide a concise (3–5 paragraph) narrative that:
1. Interprets whether the model accuracy is acceptable for financial forecasting.
2. Explains what the top features reveal about the business's revenue drivers.
3. Identifies any risks or areas where the model could fail.
4. Gives two concrete recommendations to improve forecast accuracy.

Write for a non-technical business audience. Use plain language and avoid jargon.
"""
        return self._call_with_fallback(prompt, system)

    def generate_scenario_recommendations(
        self, scenario_results_dict: Dict[str, Any]
    ) -> str:
        scenario_text = (
            pd.DataFrame(scenario_results_dict).to_string(index=False)
            if isinstance(scenario_results_dict, dict)
            else str(scenario_results_dict)
        )

        system = (
            "You are an expert financial analyst and business strategist. "
            "Provide clear, actionable, and data-driven insights. Be concise and focus on business value."
        )
        prompt = f"""You are a strategic business consultant analysing what-if scenario
simulations for a product-based business.

## Scenario Simulation Results
{scenario_text}

Analyse these results and provide:
1. **Best opportunity**: Which scenario offers the highest upside and why?
2. **Greatest risk**: Which scenario carries the most downside risk?
3. **Recommended strategy**: A balanced action plan considering all levers
   (pricing, marketing spend, demand).
4. **Implementation priorities**: Rank the three levers (price, marketing, demand)
   by expected ROI, with one-sentence rationale each.

Keep the response concise (under 400 words). Use bullet points for clarity.
"""
        return self._call_with_fallback(prompt, system)

    def generate_executive_summary(self, all_data_dict: Dict[str, Any]) -> str:
        sections: list[str] = []

        if "metrics" in all_data_dict:
            m = all_data_dict["metrics"]
            sections.append(
                f"### Model Performance\n"
                f"- R²: {m.get('R2', 'N/A')}  |  "
                f"MAPE: {m.get('MAPE', 'N/A')}%  |  "
                f"RMSE: ${m.get('RMSE', 'N/A'):,.2f}"
            )
        if "data_stats" in all_data_dict:
            sections.append(f"### Historical Data Statistics\n{all_data_dict['data_stats']}")
        if "forecast" in all_data_dict:
            sections.append(f"### 6-Month Revenue Forecast\n{all_data_dict['forecast']}")
        if "scenarios" in all_data_dict:
            sections.append(f"### Scenario Analysis Summary\n{all_data_dict['scenarios']}")
        if "feature_importance" in all_data_dict:
            sections.append(f"### Key Revenue Drivers\n{all_data_dict['feature_importance']}")

        context = "\n\n".join(sections) if sections else "Comprehensive financial data provided."

        system = (
            "You are a senior financial analyst preparing an executive briefing "
            "for the board of directors. Be authoritative yet accessible."
        )
        prompt = f"""You are a senior financial analyst preparing an executive briefing
for the board of directors of a mid-sized product company.

## Business Intelligence Data
{context}

Write a professional executive summary (4–6 paragraphs) that covers:
1. **Business Health Overview** — current performance vs expectations.
2. **Forecasting Outlook** — the 6-month revenue trajectory and confidence level.
3. **Key Drivers & Risks** — the 2–3 factors most influencing revenue.
4. **Strategic Recommendations** — three prioritised actions the leadership team
   should take in the next quarter.
5. **Conclusion** — a single sentence bottom line.

Tone: authoritative yet accessible. Suitable for a board presentation.
Avoid technical ML jargon; focus on business implications.
"""
        return self._call_with_fallback(prompt, system)

    # ------------------------------------------------------------------
    # Fallback orchestrator
    # ------------------------------------------------------------------

    def _call_with_fallback(self, prompt: str, system: str) -> str:
        """Try each provider in order; return first successful response."""
        last_error = "No providers configured."

        # 1 & 2 — Gemini keys
        for i, key in enumerate(self._gemini_keys, start=1):
            text, err = self._call_gemini(key, prompt, system)
            if text is not None:
                return text
            last_error = f"Gemini key {i}: {err}"

        # 3 — Pollinations.ai (no key needed)
        text, err = self._call_pollinations(prompt, system)
        if text is not None:
            return text
        last_error = f"Pollinations: {err}"

        # 4 — OllamaFreeAPI (no key needed)
        text, err = self._call_ollamafree(prompt, system)
        if text is not None:
            return text
        last_error = f"OllamaFree: {err}"

        # 5 — Claude
        if self._anthropic_key:
            text, err = self._call_claude(prompt, system)
            if text is not None:
                return text
            last_error = f"Claude: {err}"

        return (
            f"AI insights unavailable — all providers failed.\n"
            f"Last error: {last_error}\n\n"
            "Common causes: quota exhausted, insufficient credits, or network issue."
        )

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_gemini(
        self, api_key: str, prompt: str, system_instruction: str
    ) -> tuple[Optional[str], str]:
        try:
            from google import genai  # noqa: PLC0415
            from google.genai import types  # noqa: PLC0415

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=1024,
                ),
            )
            return response.text, ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {str(exc)[:120]}"

    def _call_pollinations(
        self, prompt: str, system_instruction: str
    ) -> tuple[Optional[str], str]:
        """Free, no key. POST to text.pollinations.ai with openai model."""
        try:
            response = requests.post(
                _POLLINATIONS_URL,
                json={
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt},
                    ],
                    "model": _POLLINATIONS_MODEL,
                    "seed": 42,
                },
                timeout=60,
            )
            response.raise_for_status()
            text = response.text.strip()
            if not text:
                return None, "Empty response"
            return text, ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {str(exc)[:120]}"

    def _call_ollamafree(
        self, prompt: str, system_instruction: str
    ) -> tuple[Optional[str], str]:
        """Free distributed Ollama nodes. No key needed."""
        try:
            import re  # noqa: PLC0415
            from ollamafreeapi import OllamaFreeAPI  # noqa: PLC0415

            client = OllamaFreeAPI()
            full_prompt = f"{system_instruction}\n\n{prompt}"
            text = client.chat(model=_OLLAMA_MODEL, prompt=full_prompt, temperature=0.3)
            if not text:
                return None, "Empty response"
            # Strip <think>...</think> reasoning blocks from models like deepseek-r1
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if not text:
                return None, "Response was only reasoning, no output"
            return text, ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {str(exc)[:120]}"

    def _call_claude(
        self, prompt: str, system_instruction: str
    ) -> tuple[Optional[str], str]:
        try:
            import anthropic  # noqa: PLC0415

            client = anthropic.Anthropic(api_key=self._anthropic_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_instruction,
                messages=[{"role": "user", "content": prompt}],
            )
            for block in response.content:
                if block.type == "text":
                    return block.text, ""
            return None, "No text block in response"
        except Exception as exc:
            return None, f"{type(exc).__name__}: {str(exc)[:120]}"
