# Midstone 2: Prepare the data (QuickPlate)

**Time:** about 2 hours · **Hand in:** 12 numbers + your code (.py or .ipynb) · **Runs on:** your laptop, with Python · **12 checks**

## The story

The app team warned Kavitha that QuickPlate's data has problems, but nobody has counted them. She wants to know **how bad they are before anyone fixes anything**.

## In short

Real data is messy. Before you fix anything, you **count what's wrong**, so you know how big each problem is. In this midstone you open one day of QuickPlate's data in Python and count its problems. **Don't fix them yet.**

**By the end, you'll have:**

- **12 numbers:** how many rows each of the 7 files has, and how many orders have each of 5 problems.
- **Your own Python code** that found them.

**Why this matters:** at work, you never clean data blind. Counting the problems first tells you which reports they would break.

## Words you'll meet

| Word | Meaning |
|---|---|
| CSV file | A text file where each line is a row and commas separate the columns. Excel can open it, but don't save it from Excel. |
| pandas | The Python library for tables. `pd.read_csv()` turns a CSV file into a DataFrame. |
| DataFrame | A table inside Python. `len(df)` tells you how many rows it has. |
| Header line | The first line of a CSV, with the column names. It isn't a data row. |
| Empty value | A cell with nothing in it. pandas shows it as `NaN`, and `isna()` finds it. |
| Duplicate | The same `order_id` appearing more than once. |

## What to do: four steps, in order

### Step 1 · Download your dataset

`midstone-02-your-data.zip`: your own day 1 data, made from your register number. 7 CSV files and nothing else.

| File | One row is |
|---|---|
| `customers.csv` | one customer |
| `restaurants.csv` | one restaurant |
| `menu_items.csv` | one dish |
| `riders.csv` | one rider |
| `orders_01.csv` | one order, day 1 |
| `order_items_01.csv` | one order line, day 1 |
| `payments_01.csv` | one payment try, day 1 |

Don't open the files in Excel and save them, because Excel quietly changes them. If that happens, download them again.

### Step 2 · Write your Python code

Open the files in VS Code or Jupyter and write Python to answer the questions below. Any approach is fine. Save your code as you go: you upload it when you hand in.

**Starting out:** `import pandas as pd`, then `orders = pd.read_csv("orders_01.csv", dtype=str)` loads a file, and `orders.head()` shows its first rows. `dtype=str` keeps every value exactly as written.

### Step 3 · Answer these questions with your code

#### Q1 · How many rows does each of the 7 files have?

| | |
|---|---|
| **Why** | So you know how much data you're working with. |
| **How** | Load each file and use `len()`. The header line isn't counted. |
| **You should see** | Seven numbers. `customers.csv` is much bigger than `restaurants.csv`. |

*Made-up example:*

```
order_id,restaurant_id,total_amount
101,R5,250
102,R7,410
103,R5,180
```

Answer: **3 rows**. The first line only holds the column names, so it isn't counted.

#### Q2 · How many order IDs appear more than once?

| | |
|---|---|
| **Why** | A repeated order would count the same sale twice. |
| **How** | In `orders_01.csv`, count how often each `order_id` appears (`value_counts()`), then count the IDs that appear more than once. An ID that's there 3 times counts as 1. |
| **You should see** | A number much smaller than the number of rows. |

*Made-up example:*

| order_id | Note |
|---|---|
| 101 | |
| 102 | 1st time |
| 102 | 2nd time |
| 103 | 1st time |
| 103 | 2nd time |
| 103 | 3rd time |

Answer: **2 IDs repeat**: 102 and 103. 103 is there 3 times, but it's one ID, so it counts once.

#### Q3 · How many orders have an unknown restaurant?

| | |
|---|---|
| **Why** | An order from a restaurant we don't know can't be given a city. |
| **How** | Two kinds, added together: `restaurant_id` is empty, **or** it isn't in `restaurants.csv` (`isin()` helps). |

*Made-up example* (`restaurants.csv` holds only R5 and R7):

| order_id | restaurant_id | Note |
|---|---|---|
| 101 | R5 | |
| 102 | (empty) | no restaurant |
| 103 | R99 | not in the list |

Answer: **2 orders**. Order 102 has no restaurant, and R99 isn't in `restaurants.csv`.

#### Q4 · How many orders have a total below 0?

| | |
|---|---|
| **Why** | A negative total would wrongly lower revenue. |
| **How** | Turn `total_amount` into a number first (`pd.to_numeric`), then count the values below 0. Exactly 0 doesn't count. |

*Made-up example:*

| order_id | total_amount | Note |
|---|---|---|
| 101 | 250 | |
| 102 | -40 | below 0 |
| 103 | 0 | exactly 0 |

Answer: **1 order**. Only order 102. A total of exactly 0 isn't below 0.

#### Q5 · How many orders write the time the other way?

| | |
|---|---|
| **Why** | Two formats in one column break any time calculation. |
| **How** | Most `order_ts` values look like `2026-03-01 14:07:00`. Some look like `01/03/2026 14:07`. Look for a character only one of them has, like `/`. |

*Made-up example:*

| order_id | order_ts |
|---|---|
| 101 | 2026-03-01 14:07:00 |
| 102 | 01/03/2026 14:07 |
| 103 | 2026-03-01 15:20:00 |

Answer: **1 order**. Only order 102 is written the other way. It has a `/`.

#### Q6 · How many orders have no customer?

| | |
|---|---|
| **Why** | Kavitha wants to know, though these orders still count. |
| **How** | Count the empty cells in `customer_id` (`isna()`). |

*Made-up example:*

| order_id | customer_id |
|---|---|
| 101 | C12 |
| 102 | (empty) |
| 103 | C40 |

Answer: **1 order**. Order 102 has no customer. It's still a real order.

### Step 4 · Hand it in

On the **Submission** tab, type your 12 numbers (the 7 row counts and the 5 other answers) and upload your code. Each number is checked against your own data right away.

## Stuck? Common mistakes

| Problem | Cause / fix |
|---|---|
| My row count is one more than expected | You probably counted the header line. `len(df)` doesn't include it. |
| IDs like 0042 turned into 42 | Read the files with `dtype=str`, so nothing is changed. |
| My repeat count is about double | You counted rows, not IDs. Count each repeated ID once. |
| Excel changed my file | Download your data again, and don't save CSV files from Excel. |

## How we check

Each number is checked against your own data, made from your register number. **Each box is marked on its own**, so a right number stays right while you fix the others. Your code isn't marked: Intelli reads it in your interview.

| # | Check |
|---|---|
| 1 | Rows in customers.csv |
| 1 | Rows in restaurants.csv |
| 1 | Rows in menu_items.csv |
| 1 | Rows in riders.csv |
| 1 | Rows in orders_01.csv |
| 1 | Rows in order_items_01.csv |
| 1 | Rows in payments_01.csv |
| 2 | Order IDs that repeat |
| 3 | Unknown restaurant |
| 4 | Total below 0 |
| 5 | Time written the other way |
| 6 | No customer |

## Rules (every midstone)

**Your own work**

- **Work on your own data.** It's made from your register number: the same kinds of problems as everyone else's, but different rows, IDs and totals. A friend's numbers won't match yours.
- **Do the work yourself.** Intelli helps you think a problem through, but never writes code for you. The interview asks about your own plan, code and numbers.

**While you build**

- **Write your own code, any way you like.** No starter code. VS Code, Jupyter or Databricks; pandas or plain Python; any file name. Save as you go: you upload it every time you hand in.
- **Don't save the data files from Excel.** It quietly changes them, and your checks compare against your original data. If you did, download your data again.

**Handing in**

- Read "How we check" before you press Check. A box left empty doesn't use a try.

**The interview**

- Take it after you pass: a short voice interview with Intelli about your own work. Answer in your own words and say why. The next midstone opens once you've taken it.
- Stay on the call screen: it runs full screen, and leaving the screen 5 times ends the call.

**Good to know**

| | |
|---|---|
| Tries | 10 per midstone, at least 2 minutes apart. An empty box, or a try that fails because of the LMS itself, isn't counted. |
| Hints | Each failed check tells you what to look at in your own work. You never see the right answer. |
| Videos | A midstone never locks the course; keep watching while you work. |
| Deadline | None. The guide time (1 to 5 hours) is a guide, not a deadline. |
| Status | Locked until the Midstone 1 interview is done. You can read everything now. |

## My answers (fill in after the task)

| # | Check | My answer |
|---|---|---|
| 1 | Rows in customers.csv | |
| 1 | Rows in restaurants.csv | |
| 1 | Rows in menu_items.csv | |
| 1 | Rows in riders.csv | |
| 1 | Rows in orders_01.csv | |
| 1 | Rows in order_items_01.csv | |
| 1 | Rows in payments_01.csv | |
| 2 | Order IDs that repeat | |
| 3 | Unknown restaurant | |
| 4 | Total below 0 | |
| 5 | Time written the other way | |
| 6 | No customer | |

**My code file:**
