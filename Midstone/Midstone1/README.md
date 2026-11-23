# Midstone 1: Scope it (QuickPlate)

**Time:** about 1 hour · **Hand in:** a short form on the Submission tab · **No code, no files**

## Background

Kavitha Raman, Head of Operations at **QuickPlate** (a food-delivery app), currently runs the business on gut feeling and spreadsheets. The apps already save every order, payment and delivery to files. She wants these turned into answers she can trust, every morning.

## Task

Before writing any code, make a **plan** for her six questions. For each question, give:

1. **The columns you need**, in `table.column` form, picked from the 8 tables below.
2. **One line on how you'd answer it**, using Kavitha's definitions.

Then add **two short notes**:

- **You won't do:** one thing that's outside the scope of this work.
- **Data can't answer:** one question the data can't answer yet.

## Kavitha's six questions

| # | Question |
|---|---|
| Q1 | How many **orders**, and how much **revenue**, does each **city** bring in? |
| Q2 | Which restaurants earn the most in each city? The **top five per city**. |
| Q3 | How long do deliveries take? The **average delivery minutes for each restaurant**. |
| Q4 | Which customers come back? Customers who order **more than once**. |
| Q5 | Where do payments fail? **Failed payments, by city**. |
| Q6 | When in the day do we earn our money? **Revenue by hour**, and how it **builds up** over the day. |

## Kavitha's words

Use these definitions so your numbers mean the same as hers.

| Word | Meaning |
|---|---|
| Delivered | An order with `status = delivered`. Cancelled orders have no `delivered_ts`. |
| Revenue | Sum of `total_amount` for **delivered orders only**. |
| Orders | Count of clean orders, **whatever their status**. |
| Delivery minutes | Minutes from `order_ts` to `delivered_ts`. Delivered orders only. |
| Late | `delivered_ts` is after `promised_ts`. |
| City of an order | The city of its **restaurant**. |
| Order hour | The hour (0–23) of `order_ts`. |

## The 8 tables

| Table | One row is | Columns |
|---|---|---|
| `customers` | A customer and their city | customer_id, full_name, city, signup_date |
| `restaurants` | A restaurant, its city, cuisine and tier | restaurant_id, name, city, cuisine, tier, opened_date |
| `menu_items` | A dish on a restaurant's menu | item_id, restaurant_id, item_name, price |
| `riders` | A delivery rider and their city | rider_id, full_name, city, joined_date |
| `orders` | An order: who, where, when, status and total | order_id, customer_id, restaurant_id, rider_id, order_ts, promised_ts, delivered_ts, status, total_amount, updated_at |
| `order_items` | One line of an order: a dish and quantity | order_id, line_no, item_id, quantity, unit_price |
| `payments` | One attempt at paying for an order | payment_id, order_id, payment_ts, method, status, amount |
| `order_events` | One step of an order (placed, delivered…) | event_id, order_id, event_type, event_ts |

## Words you'll meet

| Word | Meaning |
|---|---|
| Table | Data in rows and columns, like a spreadsheet |
| Row | One record (in `orders`, one row = one order) |
| Column | One piece of information about every row |
| ID column | A column that names each row, like `order_id`; other tables use it to point at that row |
| Join | Putting two tables together on a shared column, e.g. `orders.restaurant_id` = `restaurants.restaurant_id` |
| Metric | The number or list that answers a question |

## How to plan each question (worked example)

The example question isn't one of Kavitha's: *"How many riders does each city have?"*

1. **Find what it asks for:** riders, per city.
2. **Find the table and columns:** `riders.rider_id` to count riders, and `riders.city` for where each one works.
3. **Write one line:** "Count the riders in each city."

## How it's checked

- Only the **columns** are marked. Your written lines and notes are not.
- Each question needs the right tables and columns. A missing table **or** an unrelated extra table fails that check.
- There are 6 checks, one per question, each marked on its own.
- You get 10 tries, at least 2 minutes apart. A box left empty doesn't use a try.

**Common traps:**

- Some words, like `city`, appear in more than one table. Check Kavitha's words to see whose city is meant.
- If you don't need a column to get the number, leave it out.
- If you're told a table isn't needed, remove its columns.

## Rules

- Do the work yourself. The interview asks about your own plan.
- After passing, take the short voice interview with Intelli. It runs full screen, and leaving the screen 5 times ends the call.
- Midstone 2 (which uses the real data) opens once the interview is done.
- There's no deadline; the 1 hour is a guide.

## My plan (submitted)

**Code setup** (Midstone 2 gives the real files; names here are assumed):

```python
import pandas as pd

customers   = pd.read_csv("customers.csv")
restaurants = pd.read_csv("restaurants.csv")
orders      = pd.read_csv("orders.csv")
payments    = pd.read_csv("payments.csv")
```

All code below was tested on small sample tables. In Midstone 2, clean the data first (duplicates, missing values, text), as in the Pandas notes.

### Q1 · Orders and revenue per city

**Question:** How many orders, and how much revenue, does each city bring us?

| Table | Columns |
|---|---|
| `orders` | order_id, restaurant_id, status, total_amount |
| `restaurants` | restaurant_id, city |

**Join:** `orders.restaurant_id` = `restaurants.restaurant_id`

**How I'll answer it:**
I join the orders table and the restaurants table using restaurant_id. This gives every order its city, because Kavitha says the city of an order is the city of its restaurant. Then I group by city. For the number of orders, I count order_id for all orders, whatever their status. For revenue, I add up total_amount, but only for orders where status is delivered. I can also sort the cities by revenue to see which city brings in the most.

**Code (pandas):**

```python
df = orders.merge(restaurants, on="restaurant_id", how="left")

# Orders: all orders, whatever their status
orders_per_city = df.groupby("city")["order_id"].nunique().reset_index(name="orders")

# Revenue: delivered orders only
revenue_per_city = (df[df["status"] == "delivered"]
                    .groupby("city")["total_amount"].sum()
                    .reset_index(name="revenue"))

q1 = (orders_per_city.merge(revenue_per_city, on="city", how="left")
                     .fillna({"revenue": 0})            # a city with no delivered orders
                     .sort_values("revenue", ascending=False))
```

---

### Q2 · Top 5 restaurants per city

**Question:** Which restaurants earn the most in each city? The top five per city.

| Table | Columns |
|---|---|
| `restaurants` | restaurant_id, name, city |
| `orders` | restaurant_id, status, total_amount |

**Join:** `orders.restaurant_id` = `restaurants.restaurant_id`

**How I'll answer it:**
I join orders and restaurants using restaurant_id, so every order gets its restaurant's name and city. I keep only orders where status is delivered, because Kavitha says revenue counts delivered orders only. Then I add up total_amount for each restaurant in each city to get its revenue. I sort the restaurants in each city from highest to lowest revenue and take the top 5 per city.

**Code (pandas):**

```python
df = orders.merge(restaurants, on="restaurant_id", how="left")
delivered = df[df["status"] == "delivered"]
revenue = (delivered.groupby(["city", "restaurant_id", "name"])["total_amount"]
                    .sum().reset_index(name="revenue"))
revenue = revenue.sort_values(["city", "revenue"], ascending=[True, False])
top5 = revenue.groupby("city").head(5)
```

---

### Q3 · Average delivery time per restaurant

**Question:** How long do our deliveries take? On average, for each restaurant.

| Table | Columns |
|---|---|
| `restaurants` | restaurant_id, name |
| `orders` | restaurant_id, order_ts, delivered_ts, status |

**Join:** `orders.restaurant_id` = `restaurants.restaurant_id`

**How I'll answer it:**
I join orders and restaurants using restaurant_id, so each order gets its restaurant's name. I keep only orders where status is delivered, because Kavitha says delivery minutes count delivered orders only. For each order, I take delivered_ts minus order_ts and turn it into minutes. Then I group by restaurant_id and take the average of those minutes. That gives the average delivery time for each restaurant, and I show the restaurant name next to it.

**Code (pandas):**

```python
df = orders.merge(restaurants, on="restaurant_id", how="left")
delivered = df[df["status"] == "delivered"].copy()

delivered["order_ts"] = pd.to_datetime(delivered["order_ts"])
delivered["delivered_ts"] = pd.to_datetime(delivered["delivered_ts"])
delivered["delivery_minutes"] = (
    (delivered["delivered_ts"] - delivered["order_ts"]).dt.total_seconds() / 60)

q3 = (delivered.groupby(["restaurant_id", "name"])["delivery_minutes"]
               .mean().round(1)
               .reset_index(name="avg_delivery_minutes"))
```

**Note:** `ts` = timestamp (date + time). `order_ts` = when the order was placed, `promised_ts` = when delivery was promised, `delivered_ts` = when it actually arrived. `promised_ts` is only needed for "late", not for delivery minutes.

---

### Q4 · Customers who come back

**Question:** Which customers come back? Customers who order more than once.

| Table | Columns |
|---|---|
| `customers` | customer_id, full_name |
| `orders` | customer_id, order_id |

**Join:** `orders.customer_id` = `customers.customer_id`

**How I'll answer it:**
I join customers and orders using customer_id, so each order gets the customer's full name. I don't filter on status, because Kavitha counts orders whatever their status. I group by customer_id, not by name, because two customers can have the same name. Every order has its own order_id, so I count the unique order_ids for each customer to get their number of orders. I keep only the customers with more than 1 order. These are the customers who come back, and I show their full name next to the count.

**Code (pandas):**

```python
df = orders.merge(customers, on="customer_id", how="left")

counts = (df.groupby(["customer_id", "full_name"])["order_id"]
            .nunique()                         # unique orders per customer
            .reset_index(name="orders"))

q4 = counts[counts["orders"] > 1].sort_values("orders", ascending=False)
```

**Checker feedback:** a first try without `orders.order_id` failed with *"1 more column needed from orders: to see who orders more than once, you need each order and the customer who placed it."*

---

### Q5 · Failed payments by city

**Question:** Where do payments fail? Failed payments, by city.

| Table | Columns |
|---|---|
| `payments` | order_id, status |
| `orders` | order_id, restaurant_id |
| `restaurants` | restaurant_id, city |

**Joins:** `payments.order_id` = `orders.order_id`, then `orders.restaurant_id` = `restaurants.restaurant_id`

**How I'll answer it:**
Payments don't have a city, so I join payments to orders using order_id, and orders to restaurants using restaurant_id. This gives every payment the city of its restaurant, because Kavitha says the city of an order is the city of its restaurant. I keep only the payments where payments.status is failed. Each row in payments is one try at paying, so I group by city and count the rows to get the number of failed payments in each city.

**Code (pandas):**

```python
df = (payments
      .merge(orders[["order_id", "restaurant_id"]], on="order_id", how="left")
      .merge(restaurants[["restaurant_id", "city"]], on="restaurant_id", how="left"))

failed = df[df["status"] == "failed"]           # payments.status

q5 = (failed.groupby("city").size()             # one row = one try at paying
            .reset_index(name="failed_payments")
            .sort_values("failed_payments", ascending=False))
```

**Note:** use `payments.status` (the payment failed), not `orders.status`.

---

### Q6 · Revenue by hour, and how it builds up

**Question:** When in the day do we earn our money? Revenue by hour, and how it builds up over the day.

| Table | Columns |
|---|---|
| `orders` | order_ts, status, total_amount |

**Join:** none, one table only.

**How I'll answer it:**
I only need the orders table. From order_ts I take the hour (0 to 23), which Kavitha calls the order hour. I keep only orders where status is delivered, because revenue counts delivered orders only. I group by order hour and add up total_amount to get the revenue for each hour. Then I sort by hour from 0 to 23 and add a running total, so each hour shows all the revenue earned from the start of the day up to that hour. This shows how the money builds up over the day.

**Code (pandas):**

```python
delivered = orders[orders["status"] == "delivered"].copy()
delivered["order_hour"] = pd.to_datetime(delivered["order_ts"]).dt.hour   # 0 to 23

q6 = (delivered.groupby("order_hour")["total_amount"].sum()
               .reindex(range(24), fill_value=0)    # show all 24 hours, even empty ones
               .rename_axis("order_hour")
               .reset_index(name="revenue"))         # already sorted 0 -> 23

q6["cumulative_revenue"] = q6["revenue"].cumsum()   # how it builds up over the day
```

**Note:** "builds up" = running total (cumulative sum, `.cumsum()` in pandas). Sort by hour, not by revenue.

---

### Summary of picks

| Q | Tables | Columns |
|---|---|---|
| Q1 | orders, restaurants | orders: order_id, restaurant_id, status, total_amount · restaurants: restaurant_id, city |
| Q2 | restaurants, orders | restaurants: restaurant_id, name, city · orders: restaurant_id, status, total_amount |
| Q3 | restaurants, orders | restaurants: restaurant_id, name · orders: restaurant_id, order_ts, delivered_ts, status |
| Q4 | customers, orders | customers: customer_id, full_name · orders: customer_id, order_id |
| Q5 | payments, orders, restaurants | payments: order_id, status · orders: order_id, restaurant_id · restaurants: restaurant_id, city |
| Q6 | orders | orders: order_ts, status, total_amount |

## Two short notes

**I won't do:**
I won't build live automation for these questions. My plan answers Kavitha's six questions using the data files we already have. It won't update in real time as new orders come in; the numbers can be refreshed by running it again, for example each morning.

**Data can't answer yet:**
The data shows what happened, but not why.
- **Why deliveries take longer at certain hours.** I can see which hours are slower (using order_ts and delivered_ts), but there is no data on traffic, distance, weather or kitchen time to explain it.
- **Why an order was cancelled.** The status shows an order was cancelled, but there is no column for the reason.
- **Why a payment failed.** payments.status shows a payment failed, but not the reason (for example card declined, network error or not enough money).

## Interview prep: key points

- **City of an order = restaurant's city**, so city always comes from `restaurants.city`, never `customers.city` or `riders.city`.
- **Revenue and delivery minutes use delivered orders only; order counts use all orders**, whatever their status.
- **Group by IDs, not names**: two customers or restaurants can share a name. Show the name in the result.
- **Count the ID of what you're counting**: `order_id` for orders (Q1, Q4).
- **Payments have no city**, so Q5 needs two joins: payments → orders → restaurants.
- **"Builds up over the day"** means a running total, sorted by hour 0 → 23.