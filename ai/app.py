import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Zomato Intelligence",
    page_icon="🍽️",
    layout="wide"
)

# Import after page configuration
import dashboard
import rag_chat
import text_to_sql


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🍽️ Zomato Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Analytics",
        "🤖 Review RAG",
        "🗄️ Ask Your Data"
    ]
)


# =========================================================
# PAGE ROUTING
# =========================================================

if page == "🏠 Overview":
    dashboard.render()

elif page == "📊 Analytics":
    dashboard.render_analytics()

elif page == "🤖 Review RAG":
    rag_chat.render()

elif page == "🗄️ Ask Your Data":
    text_to_sql.render()