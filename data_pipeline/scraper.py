import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def get_soup(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )

    response.raise_for_status()

    return BeautifulSoup(
        response.text,
        "html.parser"
    )


def get_rating(book):
    rating_element = book.select_one(
        "p.star-rating"
    )

    if not rating_element:
        return None

    classes = rating_element.get("class", [])

    for rating in [
        "One",
        "Two",
        "Three",
        "Four",
        "Five"
    ]:
        if rating in classes:
            return rating

    return None


def scrape_page(page_number):
    url = (
        f"https://books.toscrape.com/catalogue/"
        f"page-{page_number}.html"
    )

    print(f"Scraping: {url}")

    soup = get_soup(url)

    books = soup.select(
        "article.product_pod"
    )

    records = []

    for book in books:

        title_element = book.select_one(
            "h3 a"
        )

        price_element = book.select_one(
            "p.price_color"
        )

        availability_element = book.select_one(
            "p.instock.availability"
        )

        if not title_element:
            continue

        title = title_element.get(
            "title",
            title_element.get_text(strip=True)
        )

        price = (
            price_element.get_text(strip=True)
            if price_element
            else None
        )

        availability = (
            availability_element.get_text(
                " ",
                strip=True
            )
            if availability_element
            else None
        )

        rating = get_rating(book)

        # The catalogue page contains category information
        # in the breadcrumb on the book detail page.
        relative_url = title_element.get("href")

        book_url = urljoin(
            url,
            relative_url
        )

        try:
            book_soup = get_soup(book_url)

            breadcrumb = book_soup.select(
                "ul.breadcrumb li"
            )

            if len(breadcrumb) >= 3:
                category = breadcrumb[-2].get_text(
                    strip=True
                )
            else:
                category = "Unknown"

        except Exception as error:

            print(
                f"Could not retrieve category "
                f"for {title}: {error}"
            )

            category = "Unknown"

        records.append(
            {
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": category
            }
        )

    return records


def scrape_catalogue(
    start_page=1,
    end_page=5
):

    all_records = []

    for page in range(
        start_page,
        end_page + 1
    ):

        try:

            records = scrape_page(page)

            all_records.extend(records)

        except Exception as error:

            print(
                f"Page {page} failed: {error}"
            )

    df = pd.DataFrame(
        all_records
    )

    return df


if __name__ == "__main__":

    df = scrape_catalogue(
        start_page=1,
        end_page=5
    )

    print("\nTotal books:", len(df))

    print(
        "Categories:",
        df["category"].nunique()
    )

    print("\nFirst 5 rows:")
    print(
        df.head().to_string(
            index=False
        )
    )

    df.to_csv(
        "raw_books.csv",
        index=False
    )

    print(
        "\nSaved raw_books.csv"
    )