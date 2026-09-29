import os

import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

CACHE_FILE = os.getenv(
    "RAG_CACHE_FILE",
    "review_embeddings.parquet"
)

CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.8-flash"
)

TOP_K = 5

SIMILARITY_THRESHOLD = 0.35


# =========================================================
# GEMINI CLIENT
# =========================================================

client = OpenAI(
    api_key=st.secrets["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# LOAD REVIEW CACHE
# =========================================================

@st.cache_data
def load_reviews():

    if not os.path.exists(CACHE_FILE):

        raise FileNotFoundError(
            f"{CACHE_FILE} was not found. "
            "Create the review embedding cache first."
        )

    df = pd.read_parquet(
        CACHE_FILE,
        engine="pyarrow"
    )

    df["embedding"] = df["embedding"].apply(
        np.asarray
    )

    return df


# =========================================================
# CREATE EMBEDDING
# =========================================================

def embed(texts):

    model = load_embedding_model()

    return model.encode(
        texts,
        normalize_embeddings=True
    )


# =========================================================
# RETRIEVE REVIEWS
# =========================================================

def find_similar_reviews(
    question,
    df
):

    question_vector = embed(
        [question]
    )[0]

    # Stack all embeddings into one matrix
    embedding_matrix = np.vstack(
        df["embedding"].to_numpy()
    )

    # Because embeddings are normalized,
    # dot product = cosine similarity
    scores = embedding_matrix @ question_vector

    results = df.copy()

    results["score"] = scores

    return results.nlargest(
        TOP_K,
        "score"
    )


# =========================================================
# RELEVANCE CHECK
# =========================================================

def is_relevant(top_reviews):

    if top_reviews.empty:

        return False

    highest_score = float(
        top_reviews["score"].iloc[0]
    )

    return highest_score >= SIMILARITY_THRESHOLD


# =========================================================
# BUILD RAG CONTEXT
# =========================================================

def build_context(top_reviews):

    context = []

    for _, row in top_reviews.iterrows():

        context.append(
            f"""
City: {row['city']}
Rating: {row['rating']} stars
Review: {row['comment']}
"""
        )

    return "\n".join(context)


# =========================================================
# ASK GEMINI
# =========================================================

def ask_llm(
    question,
    top_reviews
):

    context = build_context(
        top_reviews
    )

    system_prompt = """
You are a RAG assistant for a Zomato
customer-review dataset.

Your ONLY source of information is the
customer reviews supplied in the context.

STRICT RULES:

1. Answer ONLY using the provided reviews.

2. Do NOT use general knowledge.

3. Do NOT make up information.

4. Do NOT infer unsupported facts.

5. The dataset contains customer reviews,
cities and ratings.

6. If the reviews do not contain enough
information to answer the question, respond:

"I don't have enough information in the
available reviews to answer that."

7. If the question is completely unrelated
to Zomato customer reviews, respond:

"That question is outside the scope of
this review dataset."

8. Keep the answer concise.

9. When appropriate, mention that the
answer is based on the available customer
reviews.
"""

    user_prompt = f"""
Question:
{question}

Retrieved customer reviews:
{context}
"""

    response = client.chat.completions.create(

        model=CHAT_MODEL,

        temperature=0.1,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.choices[0].message.content


# =========================================================
# STREAMLIT PAGE
# =========================================================

def render():

    st.title("🤖 Review Intelligence")

    st.caption(
        "Ask questions grounded only in your "
        "Zomato customer reviews."
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    try:

        review_df = load_reviews()

    except Exception as e:

        st.error(str(e))

        return

    # -----------------------------------------------------
    # QUESTION
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask something about the Zomato reviews..."
    )

    if not question:

        return

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.write(question)

    # -----------------------------------------------------
    # RETRIEVAL
    # -----------------------------------------------------

    with st.spinner(
        "Searching the reviews..."
    ):

        top_reviews = find_similar_reviews(
            question,
            review_df
        )

    # -----------------------------------------------------
    # RELEVANCE
    # -----------------------------------------------------

    if not is_relevant(
        top_reviews
    ):

        with st.chat_message("assistant"):

            st.write(
                "I don't have enough information "
                "in the available reviews to answer that."
            )

        return

    # -----------------------------------------------------
    # GENERATION
    # -----------------------------------------------------

    with st.spinner(
        "Analyzing the reviews..."
    ):

        try:

            answer = ask_llm(
                question,
                top_reviews
            )

        except Exception as e:

            answer = (
                "The AI model is temporarily "
                f"unavailable: {e}"
            )

    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write(answer)