import sqlite3
import pandas as pd

DATABASE = "books.db"


def create_database(df):

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    # Enable foreign-key constraints
    cursor.execute("PRAGMA foreign_keys = ON")

    # Recreate tables
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # Categories table
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    # Books table
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # Insert categories
    categories = sorted(df["category"].unique())

    for category in categories:
        cursor.execute(
            "INSERT INTO categories (category_name) VALUES (?)",
            (category,)
        )

    # Create category -> ID mapping
    category_map = dict(
        cursor.execute(
            "SELECT category_name, category_id FROM categories"
        ).fetchall()
    )

    # Insert books
    for _, row in df.iterrows():

        cursor.execute("""
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            int(bool(row["in_stock"])),
            category_map[row["category"]]
        ))

    connection.commit()

    # Verification
    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    print(f"Database created: {DATABASE}")
    print(f"Books inserted: {book_count}")
    print(f"Categories inserted: {category_count}")

    connection.close()


if __name__ == "__main__":

    df = pd.read_csv("clean_books.csv")

    print("Loading clean_books.csv...")
    print("Rows:", len(df))

    create_database(df)