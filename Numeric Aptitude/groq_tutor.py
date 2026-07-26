"""
groq_tutor.py

Wraps the Groq API to act as the "explainer" layer of the tutor.
Important design principle (mirrors your proposal, section 5.7):
The LLM NEVER computes the answer itself. It only receives an
already-verified answer + explanation from rule_solver.py and
rephrases it in a natural, conversational, tutor-like way.
This avoids LLM arithmetic hallucination entirely.

Setup:
    pip install groq
    export GROQ_API_KEY="your_key_here"      (Linux/Mac)
    setx GROQ_API_KEY "your_key_here"        (Windows, new terminal after)

Or in Streamlit: use st.secrets["GROQ_API_KEY"] (see app.py).
"""

import os
from groq import Groq

# NOTE: Groq periodically updates/retires model names.
# Check current available models at https://console.groq.com/docs/models
# before you demo/submit -- swap this string if the model has been deprecated.
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def get_client(api_key: str | None = None) -> Groq:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("No Groq API key found. Set GROQ_API_KEY env var or pass api_key explicitly.")
    return Groq(api_key=key)


def explain_answer(client: Groq, puzzle_description: str, verified_formula: str,
                    verified_answer, raw_explanation: str, model: str = DEFAULT_MODEL) -> str:
    """
    Turns a verified, already-computed result into a friendly tutor explanation.
    The model is explicitly instructed NOT to recompute or override the answer.
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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=900,
    )
    return response.choices[0].message.content


def answer_doubt(client: Groq, chat_history: list[dict], context_snippets: list[str],
                  model: str = DEFAULT_MODEL) -> str:
    """
    Generic doubt-answering / chat turn, grounded with retrieved context
    (e.g. similar solved questions from the question bank -- this is your RAG step).
    chat_history: list of {"role": "user"/"assistant", "content": str}
    context_snippets: list of short strings retrieved from the question bank
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

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=900,
    )
    return response.choices[0].message.content


def generate_practice_question(client: Groq, weak_concept: str, difficulty: str = "medium",
                                model: str = DEFAULT_MODEL) -> str:
    """
    Generates a new practice question text targeting a specific weak concept.
    Used by the "Personalized Question Generator" module (proposal section 5.8).
    """
    prompt = (
        f"Generate ONE new aptitude practice question at {difficulty} difficulty, "
        f"specifically targeting this concept: {weak_concept}. "
        f"Provide the question, then on a new line write 'Answer:' followed by the "
        f"correct answer and a brief explanation."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content
