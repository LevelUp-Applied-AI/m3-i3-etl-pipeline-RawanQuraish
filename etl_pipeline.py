"""ETL Pipeline — Amman Digital Market Customer Analytics

Extracts data from PostgreSQL, transforms it into customer-level summaries,
validates data quality, and loads results to a database table and CSV file.
"""
from sqlalchemy import create_engine
import pandas as pd
import os


def extract(engine):
    """Extract all source tables from PostgreSQL into DataFrames.

    Args:
        engine: SQLAlchemy engine connected to the amman_market database

    Returns:
        dict: {"customers": df, "products": df, "orders": df, "order_items": df}
    """
    df_customers = pd.read_sql_table("customers", engine)
    df_products = pd.read_sql_table("products", engine)
    df_orders = pd.read_sql_table("orders", engine)
    df_order_items = pd.read_sql_table("order_items", engine)

    return {
        "customers": df_customers,
        "products": df_products,
        "orders": df_orders,
        "order_items": df_order_items
    }


def transform(data_dict):
    """Transform raw data into customer-level analytics summary.

    Steps:
    1. Join orders with order_items and products
    2. Compute line_total (quantity * unit_price)
    3. Filter out cancelled orders (status = 'cancelled')
    4. Filter out suspicious quantities (quantity > 100)
    5. Aggregate to customer level: total_orders, total_revenue,
       avg_order_value, top_category

    Args:
        data_dict: dict of DataFrames from extract()

    Returns:
        DataFrame: customer-level summary with columns:
            customer_id, customer_name, city, total_orders,
            total_revenue, avg_order_value, top_category
    """
    df_customers = data_dict["customers"]
    df_products = data_dict["products"]
    df_orders = data_dict["orders"]
    df_order_items = data_dict["order_items"]

    df = df_orders.merge(df_order_items, on="order_id") \
                  .merge(df_products, on="product_id") \
                  .merge(df_customers, on="customer_id")
    df["line_total"] = df["quantity"] * df["unit_price"]
    df = df[df["status"] != "cancelled"]
    df = df[df["quantity"] <= 100]
    summary = df.groupby(["customer_id", "customer_name", "city"]) \
                .agg(
                    total_orders=pd.NamedAgg(column="order_id", aggfunc="nunique"),
                    total_revenue=pd.NamedAgg(column="line_total", aggfunc="sum")
                ).reset_index()
    
    summary["avg_order_value"] = summary["total_revenue"] / summary["total_orders"]
    top_cat = df.groupby(["customer_id", "category"]) \
                .agg(category_revenue=pd.NamedAgg(column="line_total", aggfunc="sum")) \
                .reset_index()

    top_cat = top_cat.sort_values(["customer_id", "category_revenue"], ascending=[True, False]) \
                     .drop_duplicates(subset=["customer_id"], keep="first")
    
    summary = summary.merge(top_cat[["customer_id", "category"]], on="customer_id")
    summary = summary.rename(columns={"customer_name": "customer_name", "category": "top_category"})

    return summary



def validate(df):
    """Run data quality checks on the transformed DataFrame.

    Checks:
    - No nulls in customer_id or customer_name
    - total_revenue > 0 for all customers
    - No duplicate customer_ids
    - total_orders > 0 for all customers

    Args:
        df: transformed customer summary DataFrame

    Returns:
        dict: {check_name: bool} for each check

    Raises:
        ValueError: if any critical check fails
    """
    checks = {}
    checks["no_null_customer_id"] = df["customer_id"].notnull().all()
    checks["no_null_customer_name"] = df["customer_name"].notnull().all()
    checks["total_revenue_positive"] = (df["total_revenue"] > 0).all()
    checks["no_duplicate_customer_id"] = df["customer_id"].is_unique
    checks["total_orders_positive"] = (df["total_orders"] > 0).all()

    critical_checks = ["no_null_customer_id", "no_null_customer_name", "total_revenue_positive", "total_orders_positive"]
    for check in critical_checks:
        if not checks[check]:
            raise ValueError(f"Critical data quality check failed: {check}")

    return checks





def load(df, engine, csv_path):
    """Load customer summary to PostgreSQL table and CSV file.

    Args:
        df: validated customer summary DataFrame
        engine: SQLAlchemy engine
        csv_path: path for CSV output
    """
    df.to_sql("customer_summary", engine, if_exists="replace", index=False)
    df.to_csv(csv_path, index=False)



def main():
    """Orchestrate the ETL pipeline: extract -> transform -> validate -> load."""
    # 1. Create SQLAlchemy engine
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/amman_market"
    )
    engine = create_engine(DATABASE_URL)
    
    print("Extracting data...")
    data_dict = extract(engine)
    print("Extraction complete.")

    print("Transforming data...")
    summary_df = transform(data_dict)
    print(f"Transformation complete. {len(summary_df)} customers summarized.")

    print("Validating data...")
    checks = validate(summary_df)
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check_name}: {status}")
    print("Validation complete.")

    print("Loading data...")
    output_path = "output/customer_analytics.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    load(summary_df, engine, output_path)
    print(f"Load complete. {len(summary_df)} rows saved to database and CSV.")

if __name__ == "__main__":
    main()