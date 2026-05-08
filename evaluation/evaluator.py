"""
Evaluation Dimensions
1. Feasibility       - Are timings/logistics realistic?
2. Specificity       - Named places vs. generic suggestions?
3. Personalisation   - Does the plan match stated preferences?
4. Cultural Depth    - Local context, tips, etiquette?
5. Variety           - Non-repetitive suggestions across days?

Methods
- LLM-as-judge  : A second call scores the primary response
- Heuristic      : Rule-based checks for repetition and structural completeness
"""

import json
import re
import os
from collections import Counter
from openai import OpenAI
from prompts.system_prompts import EVALUATION_PROMPT

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


class TravelPlanEvaluator:
    """Evaluates LLM travel plan responses using LLM-as-judge + heuristics."""

    ITINERARY_KEYWORDS = [
        "morning", "afternoon", "evening", "day 1", "day 2",
        "lunch", "dinner", "breakfast", "hotel", "tip"
    ]

    def __init__(self):
        self.evaluation_history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_response(self, user_message: str, assistant_response: str,
                          conversation_history: list) -> dict:
        """
        Run full evaluation pipeline on a single assistant response.

        Returns a dict with:
          - scores       : {dimension: int} (1-5 each)
          - reasoning    : {dimension: str}
          - heuristics   : {check: bool/float}
          - overall      : str
          - is_itinerary : bool  (was this actually a travel plan response?)
        """
        is_itinerary = self._is_itinerary_response(assistant_response)

        heuristic_results = self._run_heuristics(
            assistant_response, conversation_history
        )

        if is_itinerary:
            llm_eval = self._llm_judge(user_message, assistant_response)
        else:
            # For conversational turns (e.g., clarifying questions), skip LLM eval
            llm_eval = self._default_scores(note="Non-itinerary response - LLM eval skipped")

        combined = self._combine_scores(llm_eval, heuristic_results)

        result = {
            "is_itinerary": is_itinerary,
            "scores": combined["scores"],
            "reasoning": llm_eval.get("reasoning", {}),
            "overall_feedback": llm_eval.get("overall_feedback", ""),
            "heuristics": heuristic_results,
        }

        self.evaluation_history.append(result)
        return result

    def session_summary(self, conversation_history: list) -> dict:
        """Aggregate evaluation metrics across all scored turns in a session."""
        if not self.evaluation_history:
            return {"message": "No evaluations yet in this session."}

        itinerary_evals = [e for e in self.evaluation_history if e["is_itinerary"]]
        if not itinerary_evals:
            return {"message": "No itinerary responses evaluated yet."}

        dimensions = ["feasibility", "specificity", "personalisation", "cultural_depth", "variety"]
        averages = {}
        for dim in dimensions:
            vals = [e["scores"].get(dim, 3) for e in itinerary_evals]
            averages[dim] = round(sum(vals) / len(vals), 2)

        overall_avg = round(sum(averages.values()) / len(averages), 2)

        return {
            "n_itinerary_turns": len(itinerary_evals),
            "average_scores": averages,
            "overall_average": overall_avg,
            "interpretation": self._interpret_score(overall_avg),
            "repetition_flag": any(
                e["heuristics"].get("repetition_ratio", 0) > 0.5
                for e in itinerary_evals
            )
        }

    # ------------------------------------------------------------------
    # LLM-as-judge
    # ------------------------------------------------------------------

    def _llm_judge(self, user_message: str, assistant_response: str) -> dict:
        prompt = f"""User request: {user_message}

Travel plan response to evaluate:
{assistant_response}

Score this travel plan response on the 5 dimensions."""

        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": EVALUATION_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content
            raw = re.sub(r"```json|```", "", raw).strip()
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            return self._default_scores(note=f"LLM eval failed: {str(e)}")

    # ------------------------------------------------------------------
    # Heuristic checks
    # ------------------------------------------------------------------

    def _run_heuristics(self, response: str, history: list) -> dict:
        """Fast, deterministic checks that complement the LLM judge."""
        results = {}

        # 1. Structural completeness - does it have day headers?
        day_headers = re.findall(r"###?\s*Day\s+\d", response, re.IGNORECASE)
        results["has_day_structure"] = len(day_headers) > 0
        results["n_days_detected"] = len(day_headers)

        # 2. Repetition ratio - what fraction of place-type nouns repeat?
        results["repetition_ratio"] = self._repetition_ratio(response)

        # 3. Local tip presence
        results["has_local_tips"] = bool(
            re.search(r"(💡|local tip|tip:|etiquette|note:)", response, re.IGNORECASE)
        )

        # 4. Budget mention
        results["mentions_budget"] = bool(
            re.search(r"(\$|€|£|₺|budget|price|cost|cheap|free)", response, re.IGNORECASE)
        )

        # 5. Specificity signal - count proper nouns (title-case words not at sentence start)
        proper_nouns = re.findall(r"(?<!\. )(?<!\n)[A-Z][a-z]{2,}", response)
        results["proper_noun_density"] = round(
            len(proper_nouns) / max(len(response.split()), 1), 3
        )

        # 6. Cross-turn repetition - does this response repeat places from prior assistant turns?
        prior_places = self._extract_places_from_history(history[:-2])  # exclude current turn
        current_places = set(re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", response))
        overlap = prior_places & current_places
        results["cross_turn_place_overlap"] = list(overlap)[:10]
        results["cross_turn_repetition_count"] = len(overlap)

        return results

    def _repetition_ratio(self, text: str) -> float:
        """Fraction of attraction-type words that are duplicates."""
        attraction_words = re.findall(
            r"\b(mosque|museum|palace|market|bazaar|church|temple|"
            r"restaurant|cafe|garden|park|square|tower|castle|gallery)\b",
            text.lower()
        )
        if not attraction_words:
            return 0.0
        counts = Counter(attraction_words)
        repeated = sum(v - 1 for v in counts.values() if v > 1)
        return round(repeated / len(attraction_words), 2)

    def _extract_places_from_history(self, history: list) -> set:
        """Extract capitalised place-like tokens from prior assistant messages."""
        places = set()
        for msg in history:
            if msg.get("role") == "assistant":
                found = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", msg.get("content", ""))
                places.update(found)
        return places

    def _is_itinerary_response(self, response: str) -> bool:
        """Heuristic: does this response contain a travel plan?"""
        score = 0
        for kw in self.ITINERARY_KEYWORDS:
            if kw.lower() in response.lower():
                score += 1
        return score >= 3

    # ------------------------------------------------------------------
    # Score combination
    # ------------------------------------------------------------------

    def _combine_scores(self, llm_eval: dict, heuristics: dict) -> dict:
        """
        Combine LLM judge scores with heuristic signals.
        Heuristics can nudge the variety score down if repetition is high.
        """
        scores = {
            "feasibility": llm_eval.get("feasibility", 3),
            "specificity": llm_eval.get("specificity", 3),
            "personalisation": llm_eval.get("personalisation", 3),
            "cultural_depth": llm_eval.get("cultural_depth", 3),
            "variety": llm_eval.get("variety", 3),
        }

        # Penalise variety score if heuristic detects high repetition
        rep = heuristics.get("repetition_ratio", 0)
        if rep > 0.5:
            scores["variety"] = max(1, scores["variety"] - 2)
        elif rep > 0.3:
            scores["variety"] = max(1, scores["variety"] - 1)

        # Boost specificity if high proper noun density
        if heuristics.get("proper_noun_density", 0) > 0.05:
            scores["specificity"] = min(5, scores["specificity"] + 1)

        return {"scores": scores}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_scores(self, note: str = "") -> dict:
        return {
            "feasibility": 3,
            "specificity": 3,
            "personalisation": 3,
            "cultural_depth": 3,
            "variety": 3,
            "reasoning": {},
            "overall_feedback": note,
        }

    def _interpret_score(self, score: float) -> str:
        if score >= 4.5:
            return "Excellent - highly personalised, specific, and feasible plans."
        elif score >= 3.5:
            return "Good - solid plans with some room for improvement."
        elif score >= 2.5:
            return "Fair - plans are usable but generic or have feasibility issues."
        else:
            return "Poor - plans lack specificity, personalisation, or are infeasible."
