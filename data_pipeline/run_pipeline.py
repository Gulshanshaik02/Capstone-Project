import os
import sqlite3
import pandas as pd

from scraper import scrape_catalogue
from cleaner import clean_books
from database import create_database


RAW_FILE = "raw_books.csv"
CLEAN_FILE = "clean_books.csv"
DATABASE = "books.db"
RESULT_FILE = "query_results.txt"


def run_queries():

    connection = sqlite3.connect(DATABASE)

    queries = {
        "Query 1 - SELECT + WHERE": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp > 30;
        """,

        "Query 2 - ORDER BY": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC;
        """,

        "Query 3 - LIMIT": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10;
        """,

        "Query 4 - DISTINCT": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating;
        """,

        "Query 5 - BETWEEN": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30;
        """,

        "Query 6 - IN": """
            SELECT title, rating
            FROM books
            WHERE rating IN (4, 5);
        """
    }

    with open(RESULT_FILE, "w", encoding="utf-8") as output:

        for name, query in queries.items():

            result = pd.read_sql(query, connection)

            print("\n" + "=" * 70)
            print(name)
            print("=" * 70)
            print(result.to_string(index=False))

            output.write("\n" + "=" * 70 + "\n")
            output.write(name + "\n")
            output.write("=" * 70 + "\n")
            output.write(result.to_string(index=False))
            output.write("\n")

    # SQL JOIN
    sql_join = pd.read_sql("""
        SELECT
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.title ASC
        LIMIT 10;
    """, connection)

    # pandas.merge()
    books_df = pd.read_sql("""
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            category_id
        FROM books;
    """, connection)

    categories_df = pd.read_sql("""
        SELECT
            category_id,
            category_name
        FROM categories;
    """, connection)

    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merged_df = merged_df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "category_name"
        ]
    ]

    merged_df = merged_df.sort_values(
        by=["rating", "title"],
        ascending=[False, True]
    ).head(10).reset_index(drop=True)

    sql_join = sql_join.reset_index(drop=True)

    # Compare numeric columns safely
    sql_join["price_gbp"] = sql_join["price_gbp"].astype(float).round(6)
    sql_join["price_inr"] = sql_join["price_inr"].astype(float).round(6)

    merged_df["price_gbp"] = merged_df["price_gbp"].astype(float).round(6)
    merged_df["price_inr"] = merged_df["price_inr"].astype(float).round(6)

    join_match = sql_join.equals(merged_df)

    print("\n" + "=" * 70)
    print("SQL JOIN")
    print("=" * 70)
    print(sql_join.to_string(index=False))

    print("\n" + "=" * 70)
    print("PANDAS MERGE")
    print("=" * 70)
    print(merged_df.to_string(index=False))

    print("\nSQL JOIN and pandas.merge() equivalent:", join_match)

    with open(RESULT_FILE, "a", encoding="utf-8") as output:
        output.write("\n" + "=" * 70 + "\n")
        output.write("SQL JOIN\n")
        output.write("=" * 70 + "\n")
        output.write(sql_join.to_string(index=False))

        output.write("\n\n" + "=" * 70 + "\n")
        output.write("PANDAS MERGE\n")
        output.write("=" * 70 + "\n")
        output.write(merged_df.to_string(index=False))

        output.write(
            "\n\nSQL JOIN and pandas.merge() equivalent: "
            + str(join_match)
            + "\n"
        )

    connection.close()

    return join_match


def main():

    print("\n" + "=" * 70)
    print("MODULE 1 - DATA PIPELINE")
    print("=" * 70)

    # -------------------------------------------------
    # 1. SCRAPE
    # -------------------------------------------------

    print("\n[1/4] Scraping books...")

    raw_df = scrape_catalogue(1, 5)

    if len(raw_df) < 60:
        raise ValueError(
            f"Dataset contains only {len(raw_df)} books. "
            "At least 60 are required."
        )

    raw_df.to_csv(RAW_FILE, index=False)

    print(f"Scraped books: {len(raw_df)}")
    print(f"Categories: {raw_df['category'].nunique()}")
    print(f"Saved: {RAW_FILE}")

    # -------------------------------------------------
    # 2. CLEAN
    # -------------------------------------------------

    print("\n[2/4] Cleaning data...")

    clean_df = clean_books(raw_df)

    clean_df.to_csv(CLEAN_FILE, index=False)

    print(f"Cleaned books: {len(clean_df)}")
    print(f"Saved: {CLEAN_FILE}")

    print("\nData types:")
    print(clean_df.dtypes)

    print("\nMissing values:")
    print(clean_df.isnull().sum())

    # -------------------------------------------------
    # 3. DATABASE
    # -------------------------------------------------

    print("\n[3/4] Creating SQLite database...")

    create_database(clean_df)

    # -------------------------------------------------
    # 4. QUERIES
    # -------------------------------------------------

    print("\n[4/4] Running SQL and pandas validation...")

    join_match = run_queries()

    # -------------------------------------------------
    # FINAL VALIDATION
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print("Books >= 60:", len(clean_df) >= 60)
    print("Categories >= 3:", clean_df["category"].nunique() >= 3)
    print("Rating range valid:", clean_df["rating"].between(1, 5).all())
    print("Price numeric:", pd.api.types.is_numeric_dtype(clean_df["price_gbp"]))
    print("INR conversion valid:",
          (clean_df["price_inr"] ==
           clean_df["price_gbp"] * 105.50).all())
    print("JOIN equivalence:", join_match)

    print("\nModule 1 pipeline completed successfully.")


if __name__ == "__main__":
    main()