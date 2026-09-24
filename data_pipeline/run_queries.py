import sqlite3
import pandas as pd

DATABASE = "books.db"

connection = sqlite3.connect(DATABASE)

queries = {
    "Query 1 - WHERE": """
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
    """,

    "Query 7 - JOIN": """
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
    """
}


with open("query_results.txt", "w", encoding="utf-8") as output:

    for name, query in queries.items():

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        result = pd.read_sql(query, connection)

        print(result.to_string(index=False))

        output.write("\n" + "=" * 70 + "\n")
        output.write(name + "\n")
        output.write("=" * 70 + "\n")
        output.write(result.to_string(index=False))
        output.write("\n")


# ---------------------------------------------------------
# Requirement: reproduce a JOIN using pandas.merge()
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("JOIN REPRODUCTION USING pandas.merge()")
print("=" * 70)

books_df = pd.read_sql("""
    SELECT
        book_id,
        title,
        price_gbp,
        price_inr,
        rating,
        category_id
    FROM books
""", connection)

categories_df = pd.read_sql("""
    SELECT
        category_id,
        category_name
    FROM categories
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

print(merged_df.to_string(index=False))


# ---------------------------------------------------------
# Compare SQL JOIN and pandas.merge()
# ---------------------------------------------------------

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
    ORDER BY b.rating DESC
    LIMIT 10;
""", connection)

sql_join = sql_join.reset_index(drop=True)

sql_join["price_gbp"] = sql_join["price_gbp"].round(6)
sql_join["price_inr"] = sql_join["price_inr"].round(6)

merged_df["price_gbp"] = merged_df["price_gbp"].round(6)
merged_df["price_inr"] = merged_df["price_inr"].round(6)

join_match = sql_join.equals(merged_df)

print("\nSQL JOIN and pandas.merge() equivalent:", join_match)

with open("query_results.txt", "a", encoding="utf-8") as output:
    output.write("\n\n" + "=" * 70 + "\n")
    output.write("JOIN REPRODUCTION USING pandas.merge()\n")
    output.write("=" * 70 + "\n")
    output.write(merged_df.to_string(index=False))

    output.write("\n\nSQL JOIN and pandas.merge() equivalent: ")
    output.write(str(join_match))
    output.write("\n")


connection.close()

print("\nSaved query outputs to query_results.txt")