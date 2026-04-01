"""Tests for the ETL pipeline."""

import pandas as pd
import pytest
from etl_pipeline import transform, validate


def test_transform_filters_cancelled():
    """Create test DataFrames with a cancelled order. Confirm it's excluded."""
    df_customers = pd.DataFrame({
        "customer_id": [1],
        "customer_name": ["Alice"], 
        "city": ["Amman"]
    })

    df_products = pd.DataFrame({
        "product_id": [10],
        "name": ["Widget"],
        "category": ["Gadgets"],
        "unit_price": [100]
    })

    df_orders = pd.DataFrame({
        "order_id": [100],
        "customer_id": [1],
        "order_date": ["2026-04-01"],
        "status": ["cancelled"] 
    })

    df_order_items = pd.DataFrame({
        "item_id": [1000],
        "order_id": [100],
        "product_id": [10],
        "quantity": [2]
    })

    data_dict = {
        "customers": df_customers,
        "products": df_products,
        "orders": df_orders,
        "order_items": df_order_items
    }

    summary_df = transform(data_dict)
    assert summary_df.empty, "Cancelled order should be excluded from summary"


def test_transform_filters_suspicious_quantity():
    """Create test DataFrames with quantity > 100. Confirm it's excluded."""
    df_customers = pd.DataFrame({
        "customer_id": [1],
        "customer_name": ["Alice"],  # <<< مهم هنا كمان
        "city": ["Amman"]
    })

    df_products = pd.DataFrame({
        "product_id": [10],
        "name": ["Widget"],
        "category": ["Gadgets"],
        "unit_price": [100]
    })

    df_orders = pd.DataFrame({
        "order_id": [101],
        "customer_id": [1],
        "order_date": ["2026-04-01"],
        "status": ["completed"]
    })

    df_order_items = pd.DataFrame({
        "item_id": [1001],
        "order_id": [101],
        "product_id": [10],
        "quantity": [150]  
    })

    data_dict = {
        "customers": df_customers,
        "products": df_products,
        "orders": df_orders,
        "order_items": df_order_items
    }

    summary_df = transform(data_dict)
    assert summary_df.empty, "Order with quantity > 100 should be excluded"


def test_validate_catches_nulls():
    """Create a DataFrame with null customer_id. Confirm validate() raises ValueError."""
    df = pd.DataFrame({
        "customer_id": [None],
        "customer_name": ["Alice"],
        "city": ["Amman"],
        "total_orders": [1],
        "total_revenue": [100],
        "avg_order_value": [100],
        "top_category": ["Gadgets"]
    })

    with pytest.raises(ValueError, match="Critical data quality check failed"):
        validate(df)
