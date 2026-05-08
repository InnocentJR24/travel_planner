"""
Personalized Travel Planner Chat App
NLP Assignment - LLM Application (Ollama backend)
"""

from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
import json
import uuid
from datetime import datetime
from evaluation.evaluator import TravelPlanEvaluator
from prompts.system_prompts import TRAVEL_PLANNER_SYSTEM_PROMPT
import logging
import os

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "travel-planner-nlp-assignment-2024"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

# Ollama exposes an OpenAI-compatible /v1 endpoint - no API key needed
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
evaluator = TravelPlanEvaluator()

# In-memory session store (use Redis/DB for production)
sessions = {}


@app.route("/")
def index():
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id
    sessions[session_id] = {
        "messages": [],
        "context": {},
        "created_at": datetime.now().isoformat()
    }
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = session.get("session_id")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if session_id not in sessions:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
        sessions[session_id] = {"messages": [], "context": {}, "created_at": datetime.now().isoformat()}

    sess = sessions[session_id]
    sess["messages"].append({"role": "user", "content": user_message})

    # Build messages for API call (keep last 10 turns for context)
    api_messages = sess["messages"][-20:]

    try:
        # Ollama uses the OpenAI Chat Completions format
        ollama_messages = [{"role": "system", "content": TRAVEL_PLANNER_SYSTEM_PROMPT}] + api_messages
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=ollama_messages,
        )

        assistant_reply = response.choices[0].message.content
        sess["messages"].append({"role": "assistant", "content": assistant_reply})

        # Run evaluation on the response
        eval_results = evaluator.evaluate_response(
            user_message=user_message,
            assistant_response=assistant_reply,
            conversation_history=sess["messages"]
        )

        logger.info(f"Session {session_id[:8]} | Model: {OLLAMA_MODEL} | Eval scores: {eval_results['scores']}")

        return jsonify({
            "reply": assistant_reply,
            "evaluation": eval_results,
            "session_id": session_id,
            "model": OLLAMA_MODEL,
        })

    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return jsonify({"error": f"Ollama error: {str(e)} - is Ollama running on {OLLAMA_BASE_URL}?"}), 500


@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.get("session_id")
    if session_id in sessions:
        sessions[session_id] = {
            "messages": [],
            "context": {},
            "created_at": datetime.now().isoformat()
        }
    return jsonify({"status": "reset"})


@app.route("/history", methods=["GET"])
def history():
    session_id = session.get("session_id")
    if session_id not in sessions:
        return jsonify({"messages": []})
    return jsonify({"messages": sessions[session_id]["messages"]})


@app.route("/eval_summary", methods=["GET"])
def eval_summary():
    """Return aggregated evaluation metrics for the current session."""
    session_id = session.get("session_id")
    if session_id not in sessions:
        return jsonify({"error": "No session"}), 404

    sess = sessions[session_id]
    summary = evaluator.session_summary(sess["messages"])
    return jsonify(summary)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
