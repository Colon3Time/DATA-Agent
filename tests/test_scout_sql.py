"""Tests for Scout's SQLite join logic (is_olist_db, OLIST_JOIN_QUERY, auto-detect FK)."""
import sqlite3
import unittest
import pandas as pd
from pathlib import Path


# ── helpers replicated from scout_script logic ──────────────────────────────

OLIST_JOIN_QUERY = """
SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    r.review_score,
    r.review_creation_date,
    r.review_answer_timestamp,
    oi.product_id,
    oi.seller_id,
    oi.price,
    oi.freight_value,
    op.payment_type,
    op.payment_installments,
    op.payment_value,
    p.product_category_name,
    p.product_weight_g,
    c.customer_state,
    c.customer_city
FROM orders o
JOIN order_reviews r   ON o.order_id = r.order_id
JOIN order_items oi    ON o.order_id = oi.order_id
JOIN order_payments op ON o.order_id = op.order_id AND op.payment_sequential = 1
JOIN products p        ON oi.product_id = p.product_id
JOIN customers c       ON o.customer_id = c.customer_id
"""


def is_olist_db(tables):
    required = {'orders', 'order_reviews', 'order_items', 'order_payments', 'products', 'customers'}
    return required.issubset(set(tables))


def validate_olist_output(df):
    if 'review_score' not in df.columns:
        return False, 'review_score missing'
    if df['review_score'].isna().mean() > 0.5:
        return False, 'review_score >50% NaN'
    return True, 'ok'


def detect_foreign_keys(conn, tables, sample_size=200):
    table_dfs = {t: pd.read_sql_query(f"SELECT * FROM {t} LIMIT {sample_size}", conn) for t in tables}
    fk_pairs = []
    for t1 in tables:
        for t2 in tables:
            if t1 >= t2:
                continue
            df1, df2 = table_dfs[t1], table_dfs[t2]
            for col1 in df1.columns:
                for col2 in df2.columns:
                    if col1 != col2:
                        continue
                    if 'id' not in col1.lower():
                        continue
                    vals1 = set(df1[col1].dropna().astype(str))
                    vals2 = set(df2[col2].dropna().astype(str))
                    if not vals1 or not vals2:
                        continue
                    overlap = len(vals1 & vals2) / min(len(vals1), len(vals2))
                    if overlap > 0.5:
                        fk_pairs.append((t1, t2, col1, round(overlap, 2)))
    return fk_pairs


# ── fixture builders ─────────────────────────────────────────────────────────

def make_olist_db():
    """In-memory Olist-like SQLite with minimal rows."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            order_status TEXT,
            order_purchase_timestamp TEXT,
            order_approved_at TEXT,
            order_delivered_carrier_date TEXT,
            order_delivered_customer_date TEXT,
            order_estimated_delivery_date TEXT
        );
        CREATE TABLE order_reviews (
            review_id TEXT,
            order_id TEXT,
            review_score INTEGER,
            review_creation_date TEXT,
            review_answer_timestamp TEXT
        );
        CREATE TABLE order_items (
            order_id TEXT,
            order_item_id INTEGER,
            product_id TEXT,
            seller_id TEXT,
            price REAL,
            freight_value REAL
        );
        CREATE TABLE order_payments (
            order_id TEXT,
            payment_sequential INTEGER,
            payment_type TEXT,
            payment_installments INTEGER,
            payment_value REAL
        );
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_category_name TEXT,
            product_weight_g REAL
        );
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            customer_state TEXT,
            customer_city TEXT
        );

        INSERT INTO customers VALUES ('c1','SP','Sao Paulo'),('c2','RJ','Rio');
        INSERT INTO products  VALUES ('p1','electronics',200),('p2','books',150);
        INSERT INTO orders    VALUES
            ('o1','c1','delivered','2021-01-01','2021-01-01','2021-01-03','2021-01-05','2021-01-06'),
            ('o2','c2','delivered','2021-01-02','2021-01-02','2021-01-04','2021-01-06','2021-01-07');
        INSERT INTO order_reviews VALUES ('r1','o1',5,'2021-01-06','2021-01-07'),
                                         ('r2','o2',3,'2021-01-07','2021-01-08');
        INSERT INTO order_items   VALUES ('o1',1,'p1','s1',100.0,10.0),
                                         ('o2',1,'p2','s2',50.0,5.0);
        INSERT INTO order_payments VALUES ('o1',1,'credit_card',1,110.0),
                                          ('o2',1,'boleto',2,55.0);
    """)
    return conn


def make_non_olist_db():
    """In-memory SQLite without Olist tables — triggers auto-detect FK path."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE employees (employee_id TEXT, dept_id TEXT, salary REAL, churn INTEGER);
        CREATE TABLE departments (dept_id TEXT, dept_name TEXT);
        INSERT INTO departments VALUES ('d1','Engineering'),('d2','Sales');
        INSERT INTO employees VALUES ('e1','d1',80000,0),('e2','d2',60000,1),('e3','d1',90000,0);
    """)
    return conn


# ── tests ────────────────────────────────────────────────────────────────────

class TestIsOlistDb(unittest.TestCase):
    def test_detects_olist_correctly(self):
        tables = ['orders','order_reviews','order_items','order_payments','products','customers','geolocation']
        self.assertTrue(is_olist_db(tables))

    def test_rejects_partial_olist(self):
        tables = ['orders','order_reviews','order_items']  # missing 3 required
        self.assertFalse(is_olist_db(tables))

    def test_rejects_unrelated_tables(self):
        self.assertFalse(is_olist_db(['employees','departments']))


class TestOlistJoinQuery(unittest.TestCase):
    def setUp(self):
        self.conn = make_olist_db()

    def tearDown(self):
        self.conn.close()

    def test_join_returns_expected_columns(self):
        df = pd.read_sql_query(OLIST_JOIN_QUERY, self.conn)
        self.assertIn('review_score', df.columns)
        self.assertIn('order_id', df.columns)
        self.assertIn('customer_state', df.columns)
        self.assertEqual(df.shape[1], 22)

    def test_join_returns_correct_row_count(self):
        df = pd.read_sql_query(OLIST_JOIN_QUERY, self.conn)
        # 2 orders × 1 item each × 1 payment each → 2 rows
        self.assertEqual(len(df), 2)

    def test_review_score_not_null(self):
        df = pd.read_sql_query(OLIST_JOIN_QUERY, self.conn)
        self.assertEqual(df['review_score'].isna().sum(), 0)


class TestValidateOlistOutput(unittest.TestCase):
    def _df(self, **kwargs):
        return pd.DataFrame(kwargs)

    def test_passes_when_review_score_present(self):
        df = self._df(review_score=[1, 5, 3])
        ok, msg = validate_olist_output(df)
        self.assertTrue(ok)

    def test_fails_when_review_score_missing(self):
        df = self._df(order_id=['o1'])
        ok, msg = validate_olist_output(df)
        self.assertFalse(ok)
        self.assertIn('missing', msg)

    def test_fails_when_review_score_mostly_null(self):
        import numpy as np
        df = self._df(review_score=[None, None, None, 5])
        ok, msg = validate_olist_output(df)
        self.assertFalse(ok)
        self.assertIn('NaN', msg)


class TestDetectForeignKeys(unittest.TestCase):
    def setUp(self):
        self.conn = make_non_olist_db()

    def tearDown(self):
        self.conn.close()

    def test_detects_dept_id_fk(self):
        tables = ['employees', 'departments']
        pairs = detect_foreign_keys(self.conn, tables)
        fk_cols = [col for _, _, col, _ in pairs]
        self.assertIn('dept_id', fk_cols)

    def test_overlap_above_threshold(self):
        tables = ['employees', 'departments']
        pairs = detect_foreign_keys(self.conn, tables)
        for _, _, _, overlap in pairs:
            self.assertGreater(overlap, 0.5)


class TestOlistVsNonOlist(unittest.TestCase):
    """Integration: correct path chosen based on is_olist_db()."""

    def test_olist_path_uses_template(self):
        conn = make_olist_db()
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        self.assertTrue(is_olist_db(tables))
        df = pd.read_sql_query(OLIST_JOIN_QUERY, conn)
        ok, _ = validate_olist_output(df)
        self.assertTrue(ok)
        conn.close()

    def test_non_olist_path_uses_autodetect(self):
        conn = make_non_olist_db()
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        self.assertFalse(is_olist_db(tables))
        pairs = detect_foreign_keys(conn, tables)
        self.assertTrue(len(pairs) > 0, "Auto-detect should find dept_id FK")
        conn.close()


if __name__ == "__main__":
    unittest.main()
