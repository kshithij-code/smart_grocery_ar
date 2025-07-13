import pandas as pd

# Load product data
def load_product_data(path="data/products.csv"):
    return pd.read_csv(path)

# Basic filter: highlight only items with tag == 'healthy'
def filter_products(df, filter_type="healthy"):
    if filter_type == "healthy":
        return df[df["tag"] == "healthy"]
    elif filter_type == "junk":
        return df[df["tag"] == "junk"]
    elif filter_type == "high_protein":
        return df[df["protein"] > 20]
    elif filter_type.startswith("only "):
        name = filter_type.replace("only ", "").strip().lower()
        return df[df["name"].str.lower().str.contains(name)]
    else:
        return df  # no filter
