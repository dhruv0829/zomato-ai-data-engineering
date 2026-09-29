import os

import pandas as pd
import streamlit as st
import snowflake.connector

from dotenv import load_dotenv


# =========================================================
# CONFIG
# =========================================================

load_dotenv()


# =========================================================
# SNOWFLAKE CONNECTION
# =========================================================

@st.cache_resource
def get_connection():

    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
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
# RUN QUERY
# =========================================================

def run_query(sql):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        return cursor.execute(sql).fetch_pandas_all()

    finally:
        cursor.close()


# =========================================================
# OVERVIEW
# =========================================================

def render():

    st.title("🏠 Zomato Intelligence")

    st.caption(
        "Snowflake-powered analytics and AI"
    )

    try:

        # -------------------------------------------------
        # KPI DATA
        # -------------------------------------------------

        kpi = run_query(
            """
            SELECT
                COALESCE(SUM(orders), 0) AS total_orders,
                COALESCE(SUM(gmv), 0) AS total_gmv,
                COALESCE(AVG(aov), 0) AS average_aov,
                COUNT(DISTINCT city) AS total_cities

            FROM MART_DAILY_CITY_REVENUE
            """
        )

        row = kpi.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Orders",
            f"{int(row['TOTAL_ORDERS']):,}"
        )

        col2.metric(
            "Total GMV",
            f"₹{float(row['TOTAL_GMV']):,.0f}"
        )

        col3.metric(
            "Average AOV",
            f"₹{float(row['AVERAGE_AOV']):,.0f}"
        )

        col4.metric(
            "Cities",
            int(row["TOTAL_CITIES"])
        )

        # -------------------------------------------------
        # GMV TREND
        # -------------------------------------------------

        st.subheader("GMV Trend")

        trend = run_query(
            """
            SELECT
                order_date,
                SUM(gmv) AS gmv

            FROM MART_DAILY_CITY_REVENUE

            GROUP BY order_date

            ORDER BY order_date
            """
        )

        if not trend.empty:

            trend["ORDER_DATE"] = pd.to_datetime(
                trend["ORDER_DATE"]
            )

            st.line_chart(
                trend.set_index("ORDER_DATE")["GMV"]
            )

        # -------------------------------------------------
        # TOP CITIES
        # -------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.subheader("Top Cities by GMV")

            cities = run_query(
                """
                SELECT
                    city,
                    SUM(gmv) AS gmv

                FROM MART_DAILY_CITY_REVENUE

                GROUP BY city

                ORDER BY gmv DESC

                LIMIT 10
                """
            )

            if not cities.empty:

                st.bar_chart(
                    cities.set_index("CITY")["GMV"]
                )

        # -------------------------------------------------
        # TOP RESTAURANTS
        # -------------------------------------------------

        with right:

            st.subheader("Top Restaurants")

            restaurants = run_query(
                """
                SELECT
                    restaurant_name,
                    revenue,
                    avg_customer_rating

                FROM MART_RESTAURANT_PERFORMACE

                ORDER BY revenue DESC

                LIMIT 10
                """
            )

            st.dataframe(
                restaurants,
                hide_index=True,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Could not load dashboard data: {e}"
        )


# =========================================================
# ANALYTICS
# =========================================================

def render_analytics():

    st.title("📊 Analytics")

    st.caption(
        "Explore Zomato performance from Snowflake"
    )

    try:

        # -------------------------------------------------
        # CITY FILTER
        # -------------------------------------------------

        cities = run_query(
            """
            SELECT DISTINCT city

            FROM MART_DAILY_CITY_REVENUE

            WHERE city IS NOT NULL

            ORDER BY city
            """
        )

        city_list = cities["CITY"].tolist()

        selected_city = st.selectbox(
            "Select City",
            ["All"] + city_list
        )

        # -------------------------------------------------
        # QUERY
        # -------------------------------------------------

        if selected_city == "All":

            sql = """
            SELECT
                order_date,
                SUM(orders) AS orders,
                SUM(gmv) AS gmv,
                AVG(cancel_rate) AS cancel_rate,
                AVG(aov) AS aov

            FROM MART_DAILY_CITY_REVENUE

            GROUP BY order_date

            ORDER BY order_date
            """

        else:

            safe_city = selected_city.replace(
                "'",
                "''"
            )

            sql = f"""
            SELECT
                order_date,
                SUM(orders) AS orders,
                SUM(gmv) AS gmv,
                AVG(cancel_rate) AS cancel_rate,
                AVG(aov) AS aov

            FROM MART_DAILY_CITY_REVENUE

            WHERE city = '{safe_city}'

            GROUP BY order_date

            ORDER BY order_date
            """

        df = run_query(sql)

        # -------------------------------------------------
        # TABLE
        # -------------------------------------------------

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True
        )

        # -------------------------------------------------
        # CHARTS
        # -------------------------------------------------

        if not df.empty:

            df["ORDER_DATE"] = pd.to_datetime(
                df["ORDER_DATE"]
            )

            st.subheader("Orders")

            st.line_chart(
                df.set_index("ORDER_DATE")["ORDERS"]
            )

            st.subheader("GMV")

            st.line_chart(
                df.set_index("ORDER_DATE")["GMV"]
            )

    except Exception as e:

        st.error(
            f"Could not load analytics: {e}"
        )