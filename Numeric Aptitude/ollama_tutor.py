"""
ollama_tutor.py

Replaces groq_tutor.py to match the proposal (section 5.7): the conversational
tutor must run on a LOCALLY-HOSTED open-weight model via Ollama, not a cloud API.

Same design principle as before: the LLM NEVER computes the answer itself.
It only receives an already-verified answer + explanation from rule_solver.py
and rephrases it conversationally. This avoids LLM arithmetic hallucination.

SETUP (do this on your own machine, not in a notebook/cloud environment):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model, e.g.:
           ollama pull llama3.2
       (llama3.2 is a good default -- smaller, runs on modest hardware.
        Use llama3.1:8b or mistral if you want a slightly stronger model
        and have the RAM for it.)
    3. Make sure the Ollama app/service is running in the background
       (it starts a local server at http://localhost:11434 automatically).
    4. pip install ollama
    5. Run this project's Streamlit app as normal -- no API key needed at all.

IMPORTANT: Ollama must be running locally on the SAME machine that runs
Streamlit. This will NOT work if you deploy to Streamlit Community Cloud,
since their servers can't run Ollama for you. For your professor's local
demo / your own machine, this is exactly what the proposal describes.
For a public cloud deployment, you'd need a paid GPU host running Ollama
separately, or you keep Groq as a documented fallback (see note at the
bottom of this file).
"""

import ollama

DEFAULT_MODEL = "llama3.2"


def get_client(host: str = "http://localhost:11434"):
    """Returns an Ollama client pointed at your local Ollama server."""
    return ollama.Client(host=host)


def explain_answer(client, puzzle_description: str, verified_formula: str,
                    verified_answer, raw_explanation: str, model: str = DEFAULT_MODEL) -> str:
    """
    Turns a verified, already-computed result into a friendly tutor explanation.
    Mirrors groq_tutor.explain_answer exactly, just routed through Ollama.
    """
    system_prompt = (
        "You are a friendly, encouraging aptitude-test tutor for a student preparing "
        "for corporate placement exams. You will be given a puzzle and an ALREADY VERIFIED "
        "correct rule, answer, and reasoning -- computed by a separate deterministic solver. "
        "\n\n"
        "CRITICAL: Do NOT independently analyze the numbers, do NOT compute your own "
        "differences/ratios, do NOT explore alternative patterns, and do NOT show any "
        "'let's try this... but wait...' exploration. That work has already been done "
        "correctly by the solver. Your only job is to take the verified reasoning given "
        "to you below and rephrase it in a clear, warm, step-by-step tutor voice. "
        "Treat the given formula, answer, and explanation as ground truth -- restate them, "
        "don't rediscover them. "
        "\n\n"
        "Keep your explanation under 300 words, with a clear concluding sentence. "
        "Never leave a sentence or step unfinished."
    )
    user_prompt = (
        f"Puzzle: {puzzle_description}\n"
        f"Verified rule: {verified_formula}\n"
        f"Verified answer: {verified_answer}\n"
        f"Verified reasoning (use this directly, do not redo the analysis yourself): {raw_explanation}\n\n"
        f"Rephrase the above verified reasoning for the student, in your own friendly words, "
        f"without recomputing anything."
    )

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.4, "num_predict": 500},
    )
    return response["message"]["content"]


def answer_doubt(client, chat_history: list[dict], context_snippets: list[str],
                  model: str = DEFAULT_MODEL) -> str:
    """
    Generic doubt-answering / chat turn, grounded with retrieved context
    (your RAG step -- see rag_retriever.py for the real embedding-based version).
    """
    grounding = "\n".join(f"- {s}" for s in context_snippets) if context_snippets else "None retrieved."
    system_prompt = (
        "You are an aptitude-test preparation tutor. Use the retrieved reference "
        "examples below ONLY as grounding context; do not treat them as the current "
        "question unless the student refers to them.\n\nRetrieved context:\n" + grounding +
        "\n\nKeep your answer under 300 words, with a clear concluding sentence. "
        "Never leave a sentence, list, or step unfinished -- if you're covering multiple "
        "points, budget your words up front so every point you start gets properly finished."
    )
    messages = [{"role": "system", "content": system_prompt}] + chat_history

    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": 0.5, "num_predict": 500},
    )
    return response["message"]["content"]


def generate_practice_question(client, weak_concept: str, difficulty: str = "medium",
                                model: str = DEFAULT_MODEL) -> str:
    """
    Generates a new practice question text targeting a specific weak concept.
    (Used only as a fallback for question types not covered by the synthetic
    dataset -- e.g. verbal reasoning phrasing variety. For triangle/grid/series,
    prefer sampling from dataset.csv, which is guaranteed-correct.)
    """
    prompt = (
        f"Generate ONE new aptitude practice question at {difficulty} difficulty, "
        f"specifically targeting this concept: {weak_concept}. "
        f"Provide the question, then on a new line write 'Answer:' followed by the "
        f"correct answer and a brief explanation."
    )
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 400},
    )
    return response["message"]["content"]


def check_ollama_available(host: str = "http://localhost:11434") -> tuple[bool, str]:
    """
    Quick health check used by app.py to show a clear message instead of a
    raw connection-error traceback if Ollama isn't running.
    """
    try:
        c = get_client(host)
        c.list()
        return True, "Ollama is running."
    except Exception as e:
        return False, (
            "Could not reach Ollama at " + host + ". Make sure the Ollama app/service "
            "is installed and running, and that you've pulled a model (e.g. "
            "`ollama pull llama3.2`). Error: " + str(e)
        )


# ---------------------------------------------------------------------------
# NOTE ON DEPLOYMENT vs. THE GROQ VERSION
#
# groq_tutor.py is kept in this project as an OPTIONAL fallback for public
# cloud deployment (Streamlit Community Cloud cannot run Ollama on its
# servers). If your professor asks about this, the honest, defensible
# answer is:
#   - The core, proposal-matching implementation is Ollama (local LLM),
#     used for local/offline demos exactly as proposed.
#   - Groq is an optional secondary backend, included only to make a public
#     cloud demo link possible, since Ollama can't run on free cloud hosting.
# This is a legitimate, common real-world pattern (local model for
# privacy/offline use, cloud API as a deployment convenience) and is worth
# stating explicitly in your report rather than leaving it unexplained.
# ---------------------------------------------------------------------------
