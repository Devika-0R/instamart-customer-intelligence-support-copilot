## SQL Analysis Queries

-- Q1: Find customers whose completed orders above INR 500 have a total value greater than INR 8,000.

SELECT
    customer_id,
    SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;


-- Q2: Calculate completed-order revenue by city and product category.

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders AS o
INNER JOIN customers AS c
    ON o.customer_id = c.customer_id
INNER JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY
    c.city,
    p.category
ORDER BY
    c.city,
    p.category;


-- Q3: Find products priced above the average product price.

SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4: Find customers who do not have any completed orders.

SELECT
    c.customer_id
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Find customers whose completed-order spending is greater than the average customer spending.

WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT
        AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend
FROM customer_spend AS cs
CROSS JOIN average_spend AS av
WHERE cs.total_spend > av.avg_spend
ORDER BY cs.total_spend DESC;


-- Q6: Rank customers by their total completed-order spending within each city using three window functions.

WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers AS c
    INNER JOIN orders AS o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY
        c.customer_id,
        c.city
)
SELECT
    customer_id,
    city,
    ROUND(total_spend, 2) AS total_spend,

    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_rank,

    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_rank,

    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS row_num

FROM customer_spend
ORDER BY
    city,
    spend_rank,
    customer_id;


-- Q7: Calculate daily completed-order revenue and its cumulative running total.

WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS order_date,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    order_date,
    ROUND(daily_revenue, 2) AS daily_revenue,

    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue

FROM daily_revenue
ORDER BY order_date;
