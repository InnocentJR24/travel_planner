"""
Runs the travel planner against a fixed test suite of user prompts
and records evaluation scores for reproducibility analysis.

Usage:
    python evaluation/batch_eval.py
    python evaluation/batch_eval.py --output results/run_1.json
    python evaluation/batch_eval.py --runs 3  # repeat 3x to measure variance
"""

import argparse
import json
import sys
import os
import statistics
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from evaluation.evaluator import TravelPlanEvaluator
from prompts.system_prompts import TRAVEL_PLANNER_SYSTEM_PROMPT

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
evaluator = TravelPlanEvaluator()

# ── Fixed test cases ────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "id": "TC01",
        "category": "basic_history_budget",
        "user_message": "I'm going to Istanbul for 3 days. I love history and I'm on a tight budget.",
    },
    {
        "id": "TC02",
        "category": "food_focus",
        "user_message": "Planning 2 days in Tokyo. I care mostly about food - local spots, not tourist traps. Mid-range budget.",
    },
    {
        "id": "TC03",
        "category": "refinement_less_crowded",
        "user_message": "I'm going to Rome for 4 days. I like art and history.",
        "followup": "The plan looks good but can you make it less crowded? I prefer off-the-beaten-path spots.",
    },
    {
        "id": "TC04",
        "category": "family_travel",
        "user_message": "We're taking our two kids (ages 6 and 10) to Paris for 3 days. What should we do?",
    },
    {
        "id": "TC05",
        "category": "vague_prompt",
        "user_message": "I want to travel somewhere nice in Europe for a week.",
    },
]


def run_single_test(test_case: dict) -> dict:
    """Run one test case through the LLM and evaluate."""
    messages = [{"role": "user", "content": test_case["user_message"]}]

    response = client.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "system", "content": TRAVEL_PLANNER_SYSTEM_PROMPT}] + messages,
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    eval_result = evaluator.evaluate_response(
        user_message=test_case["user_message"],
        assistant_response=reply,
        conversation_history=messages,
    )

    result = {
        "test_id": test_case["id"],
        "category": test_case["category"],
        "user_message": test_case["user_message"],
        "response_preview": reply[:300] + "..." if len(reply) > 300 else reply,
        "scores": eval_result["scores"],
        "heuristics": eval_result["heuristics"],
        "overall_feedback": eval_result["overall_feedback"],
    }

    # Handle followup turn if present
    if "followup" in test_case:
        messages.append({"role": "user", "content": test_case["followup"]})
        followup_response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": TRAVEL_PLANNER_SYSTEM_PROMPT}] + messages,
        )
        followup_reply = followup_response.choices[0].message.content
        messages.append({"role": "assistant", "content": followup_reply})

        followup_eval = evaluator.evaluate_response(
            user_message=test_case["followup"],
            assistant_response=followup_reply,
            conversation_history=messages,
        )
        result["followup_scores"] = followup_eval["scores"]

    return result


def run_batch(n_runs: int = 1, output_path: str = None):
    """Run the full test suite n_runs times for variance analysis."""
    all_runs = []

    for run_idx in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1} / {n_runs}")
        print(f"{'='*60}")
        run_results = []

        for tc in TEST_CASES:
            print(f"  Running {tc['id']}: {tc['category']}...", end=" ", flush=True)
            try:
                result = run_single_test(tc)
                run_results.append(result)
                avg = sum(result["scores"].values()) / len(result["scores"])
                print(f"avg={avg:.2f} ✓")
            except Exception as e:
                print(f"FAILED: {e}")
                run_results.append({"test_id": tc["id"], "error": str(e)})

        all_runs.append({"run": run_idx + 1, "results": run_results})

    # ── Variance analysis (if multiple runs) ──────────────────────────
    summary = compute_summary(all_runs)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": OLLAMA_MODEL,
            "n_runs": n_runs,
            "n_test_cases": len(TEST_CASES),
        },
        "runs": all_runs,
        "summary": summary,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n Results saved to {output_path}")
    else:
        print("\n── Summary ──")
        print(json.dumps(summary, indent=2))

    return output


def compute_summary(all_runs: list) -> dict:
    """Compute per-dimension mean and stdev across all runs and test cases."""
    dimensions = ["feasibility", "specificity", "personalisation", "cultural_depth", "variety"]
    dim_scores = {d: [] for d in dimensions}

    for run in all_runs:
        for result in run["results"]:
            if "scores" in result:
                for dim in dimensions:
                    if dim in result["scores"]:
                        dim_scores[dim].append(result["scores"][dim])

    stats = {}
    for dim, scores in dim_scores.items():
        if scores:
            stats[dim] = {
                "mean": round(statistics.mean(scores), 3),
                "stdev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores),
                "n": len(scores),
            }

    overall_means = [v["mean"] for v in stats.values()]
    stats["overall"] = {
        "mean": round(statistics.mean(overall_means), 3) if overall_means else 0,
        "interpretation": (
            "High variance (stdev > 0.8) suggests inconsistent LLM behaviour."
            if any(v["stdev"] > 0.8 for v in stats.values() if "stdev" in v)
            else "Low variance - LLM responses are relatively consistent across runs."
        )
    }

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate the Travel Planner LLM app")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to repeat the test suite")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    args = parser.parse_args()

    run_batch(n_runs=args.runs, output_path=args.output)
