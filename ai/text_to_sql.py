import os
import json

import pandas as pd
import streamlit as st
import snowflake.connector

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.8-flash"
)


# =========================================================
# GEMINI CLIENT
# =========================================================

client = OpenAI(
    api_key=st.secrets["GEMINI_API_KEY"],
    base_url=(
        "https://generativelanguage.googleapis.com/"
        "v1beta/openai/"
    )
)


# =========================================================
# SQL SAFETY
# =========================================================

FORBIDDEN_WORDS = [
    "drop",
    "delete",
    "truncate",
    "alter",
    "update",
    "insert",
    "create",
    "replace",
    "grant",
    "revoke",
    "merge"
]


# =========================================================
# DATABASE SCHEMA
# =========================================================

SCHEMA = """
Tables available in Snowflake.

Use bare table names only.

FCT_ORDERS(
    order_id,
    order_date,
    customer_id,
    restaurant_id,
    city,
    cuisine,
    payment_method,
    order_status,
    is_delivered,
    sales_amount,
    discount,
    delivery_fee,
    gst,
    customer_rating,
    delivery_time_min
)

DIM_RESTAURANT(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    rating,
    cost_for_two
)

DIM_CUSTOMER(
    customer_id,
    customer_name,
    age,
    age_segment,
    gender,
    city
)

MART_DAILY_CITY_REVENUNE(
    order_date,
    city,
    orders,
    cancel_rate,
    gmv,
    aov
)

MART_RESTAURANT_PERFORMACE(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    orders,
    revenue,
    avg_customer_rating,
    cancel_rate
)

MART_DELIVERY_SLA(
    city,
    order_hour,
    delivered_orders,
    p50_delivery_min,
    late_rate
)

Important:
gmv means delivered revenue.

Prefer MART tables when they directly
answer the question.
"""


# =========================================================
# SNOWFLAKE CONNECTION
# =========================================================

@st.cache_resource
def get_connection():

    return snowflake.connector.connect(

        account=os.getenv(
            "SNOWFLAKE_ACCOUNT"
        ),

        user=os.getenv(
            "SNOWFLAKE_USER"
        ),

        password=os.getenv(
            "SNOWFLAKE_PASSWORD"
        ),

        warehouse=os.getenv(
            "SNOWFLAKE_WAREHOUSE",
            "ZOMATO_WH"
        ),

        database=os.getenv(
            "SNOWFLAKE_DATABASE",
            "ZOMATO"
        ),

        schema="MARTS",

        role=os.getenv(
            "SNOWFLAKE_ROLE",
            "DBT_ROLE"
        )
    )


# =========================================================
# GENERATE SQL
# =========================================================

def generate_sql(question):

    system_prompt = f"""
You are a Snowflake SQL expert.

Write ONE read-only SQL query that answers
the user's question.

Rules:

1. SELECT or WITH queries only.

2. Never modify data.

3. Never use:
DROP
DELETE
TRUNCATE
ALTER
UPDATE
INSERT
CREATE
REPLACE
GRANT
REVOKE
MERGE

4. Use bare table names.

5. Do not use database/schema prefixes.

6. Prefer MART tables when they fit the question.

7. Add LIMIT 100 or less for list results.

8. Do not invent tables or columns.

9. Return JSON exactly in this format:

{{
    "sql": "your query"
}}

Available schema:

{SCHEMA}
"""

    response = client.chat.completions.create(

        model=MODEL,

        temperature=0,

        response_format={
            "type": "json_object"
        },

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    result = json.loads(content)

    sql = result["sql"]

    # Remove accidental prefixes
    sql = sql.replace(
        "ZOMATO.MARTS.",
        ""
    )

    sql = sql.replace(
        "ZOMATO.",
        ""
    )

    return sql.strip().rstrip(";")


# =========================================================
# SQL SAFETY CHECK
# =========================================================

def is_safe(sql):

    lowered = sql.strip().lower()

    # Must start with SELECT or WITH
    if not (
        lowered.startswith("select")
        or lowered.startswith("with")
    ):

        return False

    # Reject multiple statements
    if ";" in lowered:

        return False

    # Reject dangerous operations
    for word in FORBIDDEN_WORDS:

        if word in lowered:

            return False

    return True


# =========================================================
# RUN SQL
# =========================================================

def run_query(sql):

    conn = get_connection()

    cursor = conn.cursor()

    try:

        return cursor.execute(
            sql
        ).fetch_pandas_all()

    finally:

        cursor.close()


# =========================================================
# STREAMLIT PAGE
# =========================================================

def render():

    st.title("🗄️ Ask Your Data")

    st.caption(
        "Ask questions in English and query "
        "your Snowflake data."
    )

    # -----------------------------------------------------
    # EXAMPLES
    # -----------------------------------------------------

    with st.sidebar:

        st.subheader(
            "Example Questions"
        )

        st.markdown(
            """
            - Top 10 cities by GMV
            - Which cuisine has the most orders?
            - Average delivery time by city
            - Top restaurants by revenue
            - Cancel rate by payment method
            """
        )

    # -----------------------------------------------------
    # QUESTION
    # -----------------------------------------------------

    question = st.text_input(
        "Enter your question",

        placeholder=(
            "e.g. Top 10 restaurants "
            "by revenue in Bangalore"
        )
    )

    if not question:

        return

    # -----------------------------------------------------
    # GENERATE SQL
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Generating SQL..."
        ):

            sql = generate_sql(
                question
            )

    except Exception as e:

        st.error(
            f"Could not generate SQL: {e}"
        )

        return

    # -----------------------------------------------------
    # SAFETY
    # -----------------------------------------------------

    if not is_safe(sql):

        st.error(
            "The generated SQL failed the "
            "read-only safety check."
        )

        return

    # -----------------------------------------------------
    # SHOW SQL
    # -----------------------------------------------------

    with st.expander(
        "View generated SQL"
    ):

        st.code(
            sql,
            language="sql"
        )

    # -----------------------------------------------------
    # RUN QUERY
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Running query on Snowflake..."
        ):

            df = run_query(sql)

        st.success(
            f"{len(df)} rows returned"
        )

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )

        # -------------------------------------------------
        # SIMPLE CHART
        # -------------------------------------------------

        if (
            len(df.columns) == 2
            and pd.api.types.is_numeric_dtype(
                df.iloc[:, 1]
            )
        ):

            st.subheader(
                "Visualization"
            )

            st.bar_chart(
                df,
                x=df.columns[0],
                y=df.columns[1]
            )

    except Exception as e:

        st.error(
            f"Error running query: {e}"
        )