-- Query 1: SELECT + WHERE
SELECT title, price_gbp
FROM books
WHERE price_gbp > 30;


-- Query 2: ORDER BY
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC;


-- Query 3: LIMIT
SELECT title, price_gbp
FROM books
ORDER BY price_gbp DESC
LIMIT 10;


-- Query 4: DISTINCT
SELECT DISTINCT rating
FROM books
ORDER BY rating;


-- Query 5: BETWEEN
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 10 AND 30;


-- Query 6: IN
SELECT title, rating
FROM books
WHERE rating IN (4, 5);


-- Query 7: JOIN
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