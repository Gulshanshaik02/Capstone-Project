import pandas as pd
import numpy as np

GBP_TO_INR = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def clean_price(value):
    try:
        if pd.isna(value):
            return np.nan

        text = str(value).strip()

        # Keep only digits and decimal point
        text = "".join(
            char for char in text
            if char.isdigit() or char == "."
        )

        return float(text)

    except (ValueError, TypeError):
        return np.nan

def clean_rating(value):
    if pd.isna(value):
        return np.nan

    return RATING_MAP.get(str(value).strip(), np.nan)


def clean_stock(value):
    if pd.isna(value):
        return False

    return "in stock" in str(value).lower()


def clean_books(df):

    df = df.copy()

    # Clean price
    df["price_gbp"] = df["price"].apply(clean_price)

    # Convert star rating to integer
    df["rating"] = df["star_rating"].apply(clean_rating)

    # Convert availability to Boolean
    df["in_stock"] = df["availability"].apply(clean_stock)

    # Handle numeric parsing failures using median imputation
    for column in ["price_gbp", "rating"]:
        df[column] = df[column].fillna(df[column].median())

    df["rating"] = df["rating"].round().astype(int)

    # Convert GBP to INR using project-fixed exchange rate
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    # Keep required columns
    df = df[
        [
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "price_inr",
            "category"
        ]
    ]

    # Remove duplicate books
    df = df.drop_duplicates(
        subset=["title", "category"]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":

    print("Loading raw_books.csv...")

    raw_df = pd.read_csv("raw_books.csv")

    print("Raw dataset shape:", raw_df.shape)

    clean_df = clean_books(raw_df)

    print("\nCleaned dataset:")
    print(clean_df.head().to_string(index=False))

    print("\nCleaned shape:", clean_df.shape)

    print("\nData types:")
    print(clean_df.dtypes)

    print("\nMissing values:")
    print(clean_df.isnull().sum())

    clean_df.to_csv("clean_books.csv", index=False)

    print("\nSaved: clean_books.csv")