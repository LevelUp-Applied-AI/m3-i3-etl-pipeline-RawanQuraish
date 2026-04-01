"""ETL Pipeline — Amman Digital Market Customer Analytics with Advanced Quality Checks and Metadata"""

from sqlalchemy import create_engine
import pandas as pd
import os
import json
from datetime import datetime

def extract(engine, last_run_time=None):
    """Extract all source tables from PostgreSQL into DataFrames."""
    df_customers = pd.read_sql_table("customers", engine)
    df_products = pd.read_sql_table("products", engine)
    df_orders = pd.read_sql_table("orders", engine)
    df_order_items = pd.read_sql_table("order_items", engine)

    # Incremental ETL: filter orders newer than last_run_time
    if last_run_time is not None:
        df_orders = df_orders[df_orders["order_date"] > last_run_time]

    return {
        "customers": df_customers,
        "products": df_products,
        "orders": df_orders,
        "order_items": df_order_items
    }

def transform(data_dict):
    """Transform raw data into customer-level summary with outlier detection."""
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

    # Top category per customer
    top_cat = df.groupby(["customer_id", "category"]) \
                .agg(category_revenue=pd.NamedAgg(column="line_total", aggfunc="sum")) \
                .reset_index()
    top_cat = top_cat.sort_values(["customer_id", "category_revenue"], ascending=[True, False]) \
                     .drop_duplicates(subset=["customer_id"], keep="first")
    summary = summary.merge(top_cat[["customer_id", "category"]], on="customer_id")
    summary = summary.rename(columns={"category": "top_category"})

    # Outlier detection
    summary['is_outlier'] = summary['total_revenue'] > (summary['total_revenue'].mean() + 3*summary['total_revenue'].std())

    return summary

def validate(df):
    """Run data quality checks on the transformed DataFrame."""
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
    """Load customer summary to PostgreSQL table and CSV file."""
    df.to_sql("customer_summary", engine, if_exists="replace", index=False)
    df.to_csv(csv_path, index=False)

def main():
    """Orchestrate ETL pipeline: extract -> transform -> validate -> load."""
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/amman_market"
    )
    engine = create_engine(DATABASE_URL)

    # Get last ETL run time (for incremental loading)
    try:
        last_run_time = pd.read_sql("SELECT MAX(end_time) as last_time FROM etl_metadata", engine).iloc[0]["last_time"]
    except:
        last_run_time = None

    start_time = datetime.now()
    print("Extracting data...")
    data_dict = extract(engine, last_run_time)
    print("Extraction complete.")

    print("Transforming data...")
    summary_df = transform(data_dict)
    print(f"Transformation complete. {len(summary_df)} customers summarized.")

    print("Validating data...")
    checks = validate(summary_df)
    for check_name, passed in checks.items():
        print(f"{check_name}: {'PASS' if passed else 'FAIL'}")
    print("Validation complete.")

    print("Loading data...")
    os.makedirs("output", exist_ok=True)
    load(summary_df, engine, "output/customer_analytics.csv")

    # Generate quality report JSON
    quality_report = {
        "timestamp": datetime.now().isoformat(),
        "total_records": len(summary_df),
        "checks": {k: ("PASS" if v else "FAIL") for k,v in checks.items()},
        "outliers": summary_df[summary_df["is_outlier"]][["customer_id","total_revenue"]].to_dict(orient="records")
    }
    with open("output/quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=4)

    # Insert ETL metadata
    end_time = datetime.now()
    metadata_df = pd.DataFrame([{
        "start_time": start_time,
        "end_time": end_time,
        "rows_processed": len(summary_df),
        "status": "SUCCESS"
    }])
    metadata_df.to_sql("etl_metadata", engine, if_exists="append", index=False)

    print(f"Load complete. {len(summary_df)} rows saved to database and CSV.")
    print("Quality report saved to output/quality_report.json")
    print("ETL run metadata recorded.")

if __name__ == "__main__":
    main()