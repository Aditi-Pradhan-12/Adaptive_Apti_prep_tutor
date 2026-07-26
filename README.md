# Adaptive Aptitude Prep Tutor — Starter Build

This is a working starter version of the project covering:
- A 75,000-row synthetic pattern-puzzle dataset (triangles, grids, number series)
- The rule-induction engine (`rule_solver.py`) — reverse-engineers the hidden rule
  from solved examples using a candidate-operator search
- A simple adaptive difficulty mechanism (accuracy-based, in `app.py`)
- A basic weakness report (accuracy by puzzle type)
- Groq-powered AI tutor: explains verified answers conversationally, answers
  free-form doubts, and generates new practice questions for weak areas

Modules not yet included (from your full proposal — build these next):
- Verbal reasoning solver (syllogisms, blood relations)
- Non-verbal CNN module
- Graph/chart interpretation (OpenCV)
- Full RAG pipeline (currently a placeholder that samples random rule formulas
  as "context" — swap for real embedding-based retrieval, e.g. sentence-transformers
  + a vector store like ChromaDB, over your question bank)
- Ollama local-LLM option (currently wired to Groq's hosted API only)

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Generate the dataset (already done once, but you can regenerate anytime)

```bash
python generate_dataset.py --rows 75000 --out dataset.csv
```

You can change `--rows` to any number, or edit the rule lists in
`generate_dataset.py` to add new puzzle patterns.

## 3. Add your Groq API key

**Option A (recommended for local dev):**
```bash
export GROQ_API_KEY="your_key_here"      # Mac/Linux
setx GROQ_API_KEY "your_key_here"        # Windows (restart terminal after)
```

**Option B (for Streamlit deployment):**
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste
your key in. Never commit `secrets.toml` to GitHub — it's meant to be local/private.
If deploying on Streamlit Community Cloud, paste the same content into the app's
"Secrets" section in its dashboard settings instead.

## 4. Run the app

```bash
streamlit run app.py
```

## 5. Important note on the Groq model name

Groq periodically updates and retires model names. `groq_tutor.py` currently
uses `llama-3.3-70b-versatile` as the default — before you demo or submit,
check https://console.groq.com/docs/models for the current list of available
models and update `DEFAULT_MODEL` in `groq_tutor.py` if needed.

## 6. How the "no hallucinated math" design works

The Groq LLM is never asked to compute an answer. `rule_solver.py` computes and
verifies the answer first (deterministically); the LLM's only job is to explain
that already-correct result in natural language. This matches the design
principle in your proposal (section 5.7) and is worth explicitly mentioning in
your viva — it's the answer to "how do you stop the LLM from getting the math wrong."

## 7. Suggested next steps, in order

1. Get this base version running and demo-able end to end.
2. Add the verbal reasoning solver (syllogism → relationship graph).
3. Add real RAG: embed your question bank with sentence-transformers, store in
   ChromaDB, retrieve top-k similar questions for the chat tutor instead of the
   random-sample placeholder.
4. Add the non-verbal CNN module (synthetic rotated/mirrored shape images).
5. Swap Groq for Ollama locally if you want to demonstrate the local-LLM angle
   from your proposal too — you can keep both and let the user pick.
