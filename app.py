"""
app.py -- Adaptive Aptitude Prep Tutor (Streamlit)

Run:
    streamlit run app.py

Requires:
    dataset.csv (from generate_dataset.py) in the same folder
    GROQ_API_KEY set as an environment variable OR in .streamlit/secrets.toml as:
        GROQ_API_KEY = "your_key_here"
"""

import json
import os
import sys

# Folder reorganization support: the numeric-aptitude and verbal-reasoning
# modules now live in their own subfolders, so we add them to sys.path
# here. This keeps every "from rule_solver import ..." style import below
# working unchanged, regardless of the subfolder's name (spaces included).
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_BASE_DIR, "Numeric Aptitude"))
sys.path.append(os.path.join(_BASE_DIR, "Verbal Reasoning"))

import pandas as pd
import streamlit as st

from rule_solver import solve_3input_puzzle, solve_series
from visuals import draw_triangle_svg, draw_grid_svg, draw_series_svg
from verbal_solver import solve_verbal_question
import verbal_dataset_generator as vdg
from weakness_diagnoser import diagnose_weakness

# Primary tutor backend: Ollama (local LLM), per the project proposal.
# Groq is kept only as an optional fallback for public cloud deployment,
# since Ollama can't run on Streamlit Community Cloud's servers.
try:
    import ollama_tutor
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import groq_tutor
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(page_title="Adaptive Aptitude Prep Tutor", page_icon="🧠", layout="centered")

# ---------------------------------------------------------------------------
# Data + session state
# ---------------------------------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(os.path.join(_BASE_DIR, "Numeric Aptitude", "dataset.csv"))


def get_tutor_backend():
    """
    Returns (backend_name, client, module) -- tries Ollama first (proposal's
    primary spec), falls back to Groq only if Ollama isn't reachable and a
    Groq key is available (useful for a public cloud demo link).
    """
    if OLLAMA_AVAILABLE:
        ok, msg = ollama_tutor.check_ollama_available()
        if ok:
            return "ollama", ollama_tutor.get_client(), ollama_tutor

    if GROQ_AVAILABLE:
        key = st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
        if not key:
            import os
            key = os.environ.get("GROQ_API_KEY")
        if key:
            return "groq", groq_tutor.get_client(key), groq_tutor

    return None, None, None


if "history" not in st.session_state:
    st.session_state.history = []          # weakness tracking: list of {type, difficulty, correct}
if "chat" not in st.session_state:
    st.session_state.chat = []              # groq chat history
if "current_row" not in st.session_state:
    st.session_state.current_row = None
if "weakness_practice_row" not in st.session_state:
    st.session_state.weakness_practice_row = None

df = load_dataset()
backend_name, client, tutor_module = get_tutor_backend()

st.title("🧠 Adaptive Aptitude Prep Tutor")
st.caption("ML-based adaptive difficulty • Rule-induction engine • Local LLM tutor (Ollama)")

if client is None:
    st.warning("No tutor backend available. The quiz + rule-induction engine will still work, "
               "but AI tutor explanations are disabled until Ollama is running locally "
               "(see README.md) or a GROQ_API_KEY fallback is set.")
else:
    st.caption(f"Tutor backend in use: **{backend_name}**"
               + (" (local, offline)" if backend_name == "ollama" else " (cloud fallback)"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def pick_difficulty(puzzle_type_for_diff):
    """Simple adaptive difficulty: based on the last 5 attempts for this puzzle type."""
    recent = [h for h in st.session_state.history if h["type"] == puzzle_type_for_diff][-5:]
    if not recent:
        return "easy"
    acc = sum(h["correct"] for h in recent) / len(recent)
    if acc >= 0.8:
        return "hard"
    elif acc >= 0.5:
        return "medium"
    return "easy"


def sample_question(puzzle_type_to_sample, difficulty=None):
    """Pulls one real, verified question from the generated question bank."""
    target_diff = difficulty or pick_difficulty(puzzle_type_to_sample)
    pool = df[(df.puzzle_type == puzzle_type_to_sample) & (df.difficulty == target_diff)]
    if pool.empty:
        pool = df[df.puzzle_type == puzzle_type_to_sample]
    return pool.sample(1).iloc[0].to_dict()


def render_puzzle_and_answer(row, key_prefix):
    """
    Renders a puzzle (triangle / grid / series) with its visual, takes the
    user's answer, checks it against the rule-induction engine, logs it to
    the weakness history, and shows the Groq tutor explanation.
    key_prefix must be unique per place this is called from, so widget keys
    don't collide between the Practice Quiz tab and the Weakness Report tab.
    """
    puzzle_type = row["puzzle_type"]
    st.info(f"Difficulty: **{row['difficulty']}**")

    if puzzle_type in ("triangle", "grid"):
        examples = json.loads(row["given_examples"])
        question = json.loads(row["question"])
        key_nums = "corners" if puzzle_type == "triangle" else "grid"
        key_target = "center" if puzzle_type == "triangle" else "missing"

        st.write("**Solved examples:**")
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            with col:
                if puzzle_type == "triangle":
                    left, right, bottom = ex[key_nums]
                    svg = draw_triangle_svg(left, right, bottom, ex[key_target])
                else:
                    a, b, c = ex[key_nums]
                    svg = draw_grid_svg(a, b, c, ex[key_target])
                st.markdown(svg, unsafe_allow_html=True)

        st.write("**Now solve:**")
        if puzzle_type == "triangle":
            left, right, bottom = question[key_nums]
            svg_q = draw_triangle_svg(left, right, bottom, None, highlight_center=True)
        else:
            a, b, c = question[key_nums]
            svg_q = draw_grid_svg(a, b, c, None, highlight_d=True)
        st.markdown(svg_q, unsafe_allow_html=True)

        user_answer = st.number_input("Your answer", step=1, key=f"{key_prefix}_ans_{row['id']}")

        if st.button("Submit answer", key=f"{key_prefix}_submit_{row['id']}"):
            result = solve_3input_puzzle(examples, question[key_nums])
            correct = result["success"] and int(user_answer) == int(result["answer"])
            st.session_state.history.append({"type": puzzle_type, "difficulty": row["difficulty"], "correct": correct, "rule_id": row["rule_id"]})

            if correct:
                st.success(f"Correct! {result['explanation']}")
            else:
                st.error(f"Not quite. Correct answer: {result.get('answer', row['correct_answer'])}")
                st.write(result.get("explanation", ""))

            if client:
                with st.spinner("Tutor is explaining..."):
                    explanation = tutor_module.explain_answer(
                        client,
                        puzzle_description=f"{puzzle_type} puzzle with numbers {question[key_nums]}",
                        verified_formula=result.get("formula", row["rule_formula"]),
                        verified_answer=result.get("answer", row["correct_answer"]),
                        raw_explanation=result.get("explanation", ""),
                    )
                st.markdown("**🎓 Tutor explanation:**")
                st.write(explanation)

    elif puzzle_type == "series":
        question = json.loads(row["question"])
        st.write("**Sequence:**")
        svg_series = draw_series_svg(question["sequence"], next_val=None, highlight_next=True)
        st.markdown(svg_series, unsafe_allow_html=True)
        user_answer = st.number_input("What comes next?", step=1, key=f"{key_prefix}_ans_{row['id']}")

        if st.button("Submit answer", key=f"{key_prefix}_submit_{row['id']}"):
            result = solve_series(question["sequence"])
            correct = result["success"] and int(user_answer) == int(result["answer"])
            st.session_state.history.append({"type": puzzle_type, "difficulty": row["difficulty"], "correct": correct, "rule_id": row["rule_id"]})

            if correct:
                st.success(f"Correct! {result['explanation']}")
            else:
                st.error(f"Not quite. Correct answer: {result.get('answer', row['correct_answer'])}")
                st.write(result.get("explanation", ""))

            if client:
                with st.spinner("Tutor is explaining..."):
                    explanation = tutor_module.explain_answer(
                        client,
                        puzzle_description=f"number series {question['sequence']}",
                        verified_formula=result.get("formula", row["rule_formula"]),
                        verified_answer=result.get("answer", row["correct_answer"]),
                        raw_explanation=result.get("explanation", ""),
                    )
                st.markdown("**🎓 Tutor explanation:**")
                st.write(explanation)


tab_quiz, tab_verbal, tab_weakness, tab_tutor = st.tabs(
    ["📝 Practice Quiz", "🗣️ Verbal Reasoning", "📊 Weakness Report", "💬 AI Tutor"])

# ---------------------------------------------------------------------------
# TAB 1: Adaptive quiz
# ---------------------------------------------------------------------------
with tab_quiz:
    st.subheader("Practice a Pattern Puzzle")

    puzzle_type = st.selectbox("Puzzle type", ["triangle", "grid", "series"])

    if st.button("Get a new question", type="primary"):
        st.session_state.current_row = sample_question(puzzle_type)

    row = st.session_state.current_row
    if row:
        render_puzzle_and_answer(row, key_prefix="quiz")
    else:
        st.write("Click **Get a new question** to start.")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TAB: Verbal Reasoning (syllogisms, blood relations, direction sense, coding-decoding)
# ---------------------------------------------------------------------------
with tab_verbal:
    st.subheader("Practice Verbal Reasoning")
    st.caption("A trained classifier detects the question sub-type automatically, "
               "then a dedicated logic solver (not an LLM) computes the verified answer.")

    if "verbal_row" not in st.session_state:
        st.session_state.verbal_row = None

    verbal_subtype = st.selectbox(
        "Sub-type", ["blood_relation", "direction_sense", "coding_decoding", "syllogism"]
    )

    if st.button("Get a new verbal question", type="primary"):
        gen_map = {
            "blood_relation": vdg.gen_blood_relation_row,
            "direction_sense": vdg.gen_direction_row,
            "coding_decoding": vdg.gen_coding_row,
            "syllogism": vdg.gen_syllogism_row,
        }
        st.session_state.verbal_row = gen_map[verbal_subtype]()

    vrow = st.session_state.verbal_row
    if vrow:
        st.write(f"**Question:** {vrow['question']}")

        user_verbal_answer = st.text_input("Your answer", key=f"verbal_ans_{vrow['question'][:20]}")

        if st.button("Submit", key="verbal_submit"):
            result = solve_verbal_question(vrow["question"])
            correct = result["success"] and user_verbal_answer.strip().lower() == str(result["answer"]).strip().lower()

            st.session_state.history.append({
                "type": f"verbal_{vrow['subtype']}", "difficulty": "n/a",
                "correct": correct, "rule_id": f"verbal_{vrow['subtype']}"
            })

            if result["success"]:
                st.info(f"Detected sub-type: **{result['detected_subtype']}**")
                if correct:
                    st.success(f"Correct! {result['explanation']}")
                else:
                    st.error(f"Not quite. Correct answer: {result['answer']}")
                    st.write(result["explanation"])

                if client:
                    with st.spinner("Tutor is explaining..."):
                        tutor_explanation = tutor_module.explain_answer(
                            client,
                            puzzle_description=vrow["question"],
                            verified_formula=result["detected_subtype"],
                            verified_answer=result["answer"],
                            raw_explanation=result["explanation"],
                        )
                    st.markdown("**🎓 Tutor explanation:**")
                    st.write(tutor_explanation)
            else:
                st.warning(result["message"])
    else:
        st.write("Click **Get a new verbal question** to start.")

# ---------------------------------------------------------------------------
# TAB: Weakness Report
# ---------------------------------------------------------------------------
with tab_weakness:
    st.subheader("Your Performance by Puzzle Type")
    if not st.session_state.history:
        st.write("No attempts yet — solve a few questions in the Practice tab first.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        summary = hist_df.groupby("type")["correct"].agg(["mean", "count"]).rename(
            columns={"mean": "accuracy", "count": "attempts"})
        summary["accuracy"] = (summary["accuracy"] * 100).round(1)
        st.dataframe(summary)
        st.bar_chart(summary["accuracy"])

        weakest = summary["accuracy"].idxmin()
        st.write(f"📌 Weakest area right now (by puzzle type): **{weakest}**")

        st.markdown("---")
        st.subheader("Concept-Level Weakness Diagnosis")
        st.caption("Uses Sentence-BERT embeddings + KMeans clustering on your incorrect "
                   "answers to find the underlying concepts causing mistakes -- grouping "
                   "related rules together even across puzzle types, not just a flat "
                   "per-type score.")

        wrong_rule_ids = [h["rule_id"] for h in st.session_state.history if not h["correct"]]

        if not wrong_rule_ids:
            st.write("No incorrect answers yet to diagnose -- nice work so far!")
        else:
            with st.spinner("Clustering your mistakes by concept..."):
                clusters = diagnose_weakness(wrong_rule_ids)

            for i, cluster in enumerate(clusters, 1):
                st.markdown(f"**Concept group {i}** — {cluster['count']} mistake(s), "
                           f"rules involved: {', '.join(cluster['concepts'])}")
                st.write(cluster["description"])
                st.write("")

        if st.button("Practice my weakest area now"):
            if weakest.startswith("verbal_"):
                st.info("Your weakest area is a verbal reasoning type — head over to the "
                       "**Verbal Reasoning** tab and select that sub-type to practice it.")
                st.session_state.weakness_practice_row = None
            else:
                st.session_state.weakness_practice_row = sample_question(weakest)

        if st.session_state.weakness_practice_row:
            st.markdown("---")
            render_puzzle_and_answer(st.session_state.weakness_practice_row, key_prefix="weak")

# ---------------------------------------------------------------------------
# TAB 3: Free-form AI tutor chat
# ---------------------------------------------------------------------------
with tab_tutor:
    st.subheader("Ask the AI Tutor")
    if client is None:
        st.write("Set your GROQ_API_KEY to enable the chat tutor.")
    else:
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_msg = st.chat_input("Ask a doubt, e.g. 'How do I solve number-in-triangle puzzles?'")
        if user_msg:
            st.session_state.chat.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)

            # Simple retrieval placeholder: pull 2 random rule formulas as "context"
            # (swap this for real embedding-based retrieval per your proposal's RAG design)
            context = df["rule_formula"].drop_duplicates().sample(3).tolist()

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    reply = tutor_module.answer_doubt(client, st.session_state.chat, context)
                st.write(reply)
            st.session_state.chat.append({"role": "assistant", "content": reply})
