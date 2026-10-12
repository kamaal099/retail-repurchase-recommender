"""Business KPI queries run in SQL (DuckDB) over the cleaned sales table."""
import duckdb
import pandas as pd

QUERIES = {
    "monthly_revenue": """
        SELECT date_trunc('month', invoice_date) AS month,
               ROUND(SUM(revenue), 2) AS revenue,
               COUNT(DISTINCT invoice) AS orders,
               COUNT(DISTINCT customer_id) AS customers
        FROM sales
        GROUP BY 1
        ORDER BY 1
    """,
    "top_products": """
        SELECT stock_code,
               ANY_VALUE(description) AS description,
               ROUND(SUM(revenue), 2) AS revenue,
               SUM(quantity) AS units
        FROM sales
        GROUP BY stock_code
        ORDER BY revenue DESC
        LIMIT 20
    """,
    "country_revenue": """
        SELECT country,
               ROUND(SUM(revenue), 2) AS revenue,
               COUNT(DISTINCT customer_id) AS customers
        FROM sales
        GROUP BY country
        ORDER BY revenue DESC
    """,
    "repeat_customer_rate": """
        WITH per_customer AS (
            SELECT customer_id, COUNT(DISTINCT invoice) AS orders FROM sales GROUP BY 1
        )
        SELECT COUNT(*) AS customers,
               SUM(CASE WHEN orders > 1 THEN 1 ELSE 0 END) AS repeat_customers,
               ROUND(AVG(CASE WHEN orders > 1 THEN 1.0 ELSE 0.0 END), 4) AS repeat_rate
        FROM per_customer
    """,
}


def run_kpi_queries(sales: pd.DataFrame) -> dict[str, pd.DataFrame]:
    con = duckdb.connect()
    try:
        con.register("sales", sales)
        return {name: con.execute(sql).df() for name, sql in QUERIES.items()}
    finally:
        con.close()