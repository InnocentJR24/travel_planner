# Personalised Travel Planner

A conversational LLM application that generates personalised travel itineraries,
supports interactive refinement, and evaluates its own output quality.

## Project Structure

```
travel_planner/
├── app.py                     # Flask web server + API routes
├── requirements.txt
├── .env.example
├── prompts/
│   ├── __init__.py
│   └── system_prompts.py      # System prompt + evaluation prompt (with design rationale)
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py           # TravelPlanEvaluator: LLM-as-judge + heuristics
│   └── batch_eval.py          # Reproducibility / batch evaluation script
└── templates/
    └── index.html             # Chat UI with live evaluation sidebar
```

## Setup

```bash
# 1. Install & start Ollama  (https://ollama.com)
ollama pull llama3.2        # or mistral, gemma3, phi4, etc.
ollama serve                # runs on http://localhost:11434

# 2. Install Python deps
cd travel_planner
pip install -r requirements.txt

# 3. (Optional) override model or URL via env
export OLLAMA_MODEL=mistral
export OLLAMA_BASE_URL=http://localhost:11434/v1

# 4. Run
python app.py               # → http://localhost:5000
```

Ollama exposes an OpenAI-compatible `/v1` endpoint, so the app uses the
`openai` Python SDK pointed at `localhost:11434` - no API key required.

### Batch Evaluation for Reproducibility

```bash
# Run fixed test suite once
python evaluation/batch_eval.py --output results/run_1.json

# Run 3 times to measure variance (stdev across runs)
python evaluation/batch_eval.py --runs 3 --output results/variance_study.json
```
