"""
Scout Join Logic & Rows Ratio Gate Tests
ครอบคลุม 4 จุดที่ขาด:
  1. validate_output() rows ratio gate (FAIL/WARN/PASS zones + boundaries)
  2. build_joined_dataset() base table selection (Priority 1/2/3 + geo skip)
  3. build_joined_dataset() join result (rows, cols, column rename)
  4. FK edge cases (overlap=50%, zero FK, empty table, no id col)
  5. Forbidden keyword typo _lenght
"""
import sqlite3
import unittest

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Functions replicated from KB spec (scout.md) — canonical source
# ═══════════════════════════════════════════════════════════════════════

FORBIDDEN_TARGET_SUFFIXES = (
    '_cm', '_g', '_mm', '_kg', '_lb',
    '_lenght', '_length', '_width', '_height',
    '_lat', '_lng', '_latitude', '_longitude',
    '_zip', '_prefix', '_code',
)
FORBIDDEN_TARGET_KEYWORDS = {
    'zip_code', 'zip_prefix', 'geolocation', 'latitude', 'longitude',
    'product_id', 'order_id', 'customer_id', 'seller_id', 'review_id',
    'product_name_lenght', 'product_description_lenght',
    'product_weight_g', 'product_length_cm', 'product_height_cm',
    'product_width_cm', 'product_photos_qty',
}
FACT_TABLE_PRIORITY = [
    'orders', 'order', 'transactions', 'transaction', 'sales', 'sale',
    'employees', 'employee', 'staff',
    'payments', 'payment', 'invoices', 'invoice',
    'patients', 'patient', 'visits', 'visit',
    'facts', 'fact', 'events', 'event', 'logs', 'log',
]
SKIP_TABLES = ['geolocation', 'geo', 'zip', 'postal', 'translation', 'category']


def is_forbidden_target(col):
    col_l = col.lower()
    if col_l in {k.lower() for k in FORBIDDEN_TARGET_KEYWORDS}:
        return True
    if any(col_l.endswith(s) for s in FORBIDDEN_TARGET_SUFFIXES):
        return True
    if col_l.endswith('_id') or col_l.startswith('id_'):
        return True
    return False


def validate_output(df_out, sizes):
    """General rows ratio gate (from scout.md spec)."""
    largest = max(sizes.values())
    n = len(df_out)
    if n < largest * 0.1:
        return False, 'GATE FAIL'
    if n < largest * 0.5:
        return True, 'WARN'
    return True, 'GATE PASS'


def get_table_sizes(conn, tables):
    return {t: pd.read_sql_query(f"SELECT COUNT(*) as n FROM {t}", conn).iloc[0, 0]
            for t in tables}


def detect_foreign_keys(conn, tables, sample_size=200):
    table_dfs = {t: pd.read_sql_query(f"SELECT * FROM {t} LIMIT {sample_size}", conn)
                 for t in tables}
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
                    if ('id' not in col1.lower() and
                            col1 not in ['order_id', 'customer_id', 'seller_id', 'product_id', 'review_id']):
                        continue
                    vals1 = set(df1[col1].dropna().astype(str))
                    vals2 = set(df2[col2].dropna().astype(str))
                    if not vals1 or not vals2:
                        continue
                    overlap = len(vals1 & vals2) / min(len(vals1), len(vals2))
                    if overlap > 0.5:
                        fk_pairs.append((t1, t2, col1, round(overlap, 2)))
    return fk_pairs


def build_joined_dataset(conn, tables, fk_pairs):
    """Join ตาม FK — Priority 1 exact → Priority 2 keyword → Priority 3 fallback largest."""
    sizes = get_table_sizes(conn, tables)

    fk_tables = set()
    for t1, t2, col, _ in fk_pairs:
        fk_tables.add(t1)
        fk_tables.add(t2)

    if not fk_tables:
        base_table = max(sizes, key=sizes.get)
        return pd.read_sql_query(f"SELECT * FROM {base_table}", conn), base_table

    # Priority 1 — exact name match
    base_table = None
    for pname in FACT_TABLE_PRIORITY:
        for t in tables:
            if t.lower() == pname and t in fk_tables:
                base_table = t
                break
        if base_table:
            break

    # Priority 2 — keyword contained in table name
    if not base_table:
        for pname in FACT_TABLE_PRIORITY:
            for t in fk_tables:
                if pname in t.lower() and t.lower() != 'geolocation':
                    base_table = t
                    break
            if base_table:
                break

    # Priority 3 — largest non-skip table
    if not base_table:
        eligible = {t: s for t, s in sizes.items()
                    if t in fk_tables and not any(skip in t.lower() for skip in SKIP_TABLES)}
        base_table = (max(eligible, key=eligible.get) if eligible
                      else max(fk_tables, key=lambda t: sizes.get(t, 0)))

    df_base = pd.read_sql_query(f"SELECT * FROM {base_table}", conn)
    joined = {base_table}
    for t1, t2, col, _ in sorted(fk_pairs, key=lambda x: -x[3]):
        other  = t2 if t1 in joined else t1
        anchor = t1 if other == t2 else t2
        if other in joined or anchor not in joined:
            continue
        try:
            df_other = pd.read_sql_query(f"SELECT * FROM {other}", conn)
            rename = {c: f"{other}_{c}" for c in df_other.columns
                      if c in df_base.columns and c != col}
            df_other = df_other.rename(columns=rename)
            df_base  = df_base.merge(df_other, on=col, how='left')
            joined.add(other)
        except Exception:
            pass

    return df_base, base_table


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

def make_employees_db():
    """employees + departments — Priority 1 exact match 'employees'."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE employees   (employee_id TEXT, dept_id TEXT, salary REAL, churn INTEGER);
        CREATE TABLE departments (dept_id TEXT, dept_name TEXT);
        INSERT INTO departments VALUES ('d1','Engineering'),('d2','Sales'),('d3','HR');
        INSERT INTO employees   VALUES
            ('e1','d1',80000,0), ('e2','d2',60000,1), ('e3','d1',90000,0),
            ('e4','d3',70000,0), ('e5','d2',65000,1);
    """)
    return conn


def make_user_orders_db():
    """user_orders + users — Priority 2 keyword 'order' in name."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE user_orders (order_id TEXT, user_id TEXT, amount REAL);
        CREATE TABLE users       (user_id TEXT, email TEXT);
        INSERT INTO users       VALUES ('u1','a@x.com'),('u2','b@x.com'),('u3','c@x.com');
        INSERT INTO user_orders VALUES
            ('o1','u1',100),('o2','u2',200),('o3','u3',150),
            ('o4','u1',80), ('o5','u2',300),('o6','u3',120);
    """)
    return conn


def make_fallback_geo_db():
    """main_data + geo_lookup + dim_ref — Priority 3 fallback, geo skipped."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE main_data  (item_id TEXT, ref_id TEXT, value REAL);
        CREATE TABLE geo_lookup (ref_id TEXT, region TEXT);
        CREATE TABLE dim_ref    (ref_id TEXT, label TEXT);
        INSERT INTO dim_ref    VALUES ('r1','LabelA'),('r2','LabelB'),('r3','LabelC');
        INSERT INTO geo_lookup VALUES ('r1','N'),('r2','S'),('r3','E'),('r4','W'),('r5','C');
    """)
    for i in range(10):
        conn.execute(f"INSERT INTO main_data VALUES ('i{i}','r{(i%3)+1}',{i*10.0})")
    conn.commit()
    return conn


def make_duplicate_col_db():
    """purchases + users — 'category' in both → users.category renamed to users_category."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE purchases (user_id TEXT, item_id TEXT, category TEXT, amount REAL);
        CREATE TABLE users     (user_id TEXT, category TEXT, city TEXT);
        INSERT INTO users     VALUES ('u1','VIP','BKK'),('u2','Normal','CNX'),('u3','VIP','HKT');
        INSERT INTO purchases VALUES
            ('u1','p1','Electronics',500),
            ('u2','p2','Books',150),
            ('u3','p3','Electronics',800);
    """)
    return conn


def make_no_id_db():
    """tables without any id column — no FK detected → fallback to largest."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE alpha (name TEXT, age INTEGER, salary REAL);
        CREATE TABLE beta  (dept TEXT, location TEXT);
        INSERT INTO alpha VALUES ('A',30,50000),('B',25,40000),('C',35,60000),
                                 ('D',28,45000),('E',32,55000);
        INSERT INTO beta  VALUES ('Eng','BKK'),('Sales','CNX');
    """)
    return conn


def make_overlap_50pct_db():
    """overlap = exactly 50% — must NOT be detected (rule: > 0.5, not >=)."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE t1 (user_id TEXT, val INTEGER);
        CREATE TABLE t2 (user_id TEXT, city TEXT);
        INSERT INTO t1 VALUES ('u1',10),('u2',20);
        INSERT INTO t2 VALUES ('u2','BKK'),('u3','CNX');
    """)
    # overlap = {u2} / min(2,2) = 1/2 = 0.50 → NOT > 0.50
    return conn


def make_overlap_above_50pct_db():
    """overlap = 2/3 = 66.7% — must be detected."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE t1 (user_id TEXT, score INTEGER);
        CREATE TABLE t2 (user_id TEXT, city TEXT);
        INSERT INTO t1 VALUES ('u1',10),('u2',20),('u3',30);
        INSERT INTO t2 VALUES ('u2','BKK'),('u3','CNX'),('u4','HKT');
    """)
    # overlap = {u2,u3} / min(3,3) = 2/3 = 0.667 → detected
    return conn


def make_empty_table_db():
    """one table is empty — FK detection must not crash."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE facts      (record_id TEXT, dept_id TEXT, val REAL);
        CREATE TABLE empty_dim  (dept_id TEXT, name TEXT);
        INSERT INTO facts VALUES ('r1','d1',100),('r2','d2',200);
    """)
    return conn


def _get_tables(conn):
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]


# ═══════════════════════════════════════════════════════════════════════
# 1. VALIDATE OUTPUT — rows ratio gate
# ═══════════════════════════════════════════════════════════════════════

class TestValidateOutputGate(unittest.TestCase):

    def _df(self, n):
        return pd.DataFrame({'x': range(n)})

    # ── GATE FAIL zone (<10%) ──

    def test_fail_at_9pct(self):
        ok, label = validate_output(self._df(9), {'main': 100})
        self.assertFalse(ok)
        self.assertEqual(label, 'GATE FAIL')

    def test_fail_at_zero_rows(self):
        ok, _ = validate_output(self._df(0), {'main': 100})
        self.assertFalse(ok)

    def test_fail_at_1_row_of_1000(self):
        ok, _ = validate_output(self._df(1), {'main': 1000})
        self.assertFalse(ok)

    # ── boundary: exactly 10% → WARN (not FAIL) ──

    def test_boundary_10pct_is_warn_not_fail(self):
        ok, label = validate_output(self._df(10), {'main': 100})
        self.assertTrue(ok)
        self.assertEqual(label, 'WARN')

    # ── WARN zone (10%–49%) ──

    def test_warn_at_25pct(self):
        ok, label = validate_output(self._df(25), {'main': 100})
        self.assertTrue(ok)
        self.assertEqual(label, 'WARN')

    def test_warn_at_49pct(self):
        ok, label = validate_output(self._df(49), {'main': 100})
        self.assertTrue(ok)
        self.assertEqual(label, 'WARN')

    # ── boundary: exactly 50% → PASS (not WARN) ──

    def test_boundary_50pct_is_pass(self):
        ok, label = validate_output(self._df(50), {'main': 100})
        self.assertTrue(ok)
        self.assertEqual(label, 'GATE PASS')

    # ── GATE PASS zone (≥50%) ──

    def test_pass_at_100pct(self):
        ok, label = validate_output(self._df(100), {'main': 100})
        self.assertTrue(ok)
        self.assertEqual(label, 'GATE PASS')

    def test_pass_above_100pct(self):
        # one-to-many join ขยาย rows ได้
        ok, label = validate_output(self._df(150), {'main': 100})
        self.assertTrue(ok)
        self.assertEqual(label, 'GATE PASS')

    def test_uses_largest_table_for_ratio(self):
        # largest=500, output=40 → 8% < 10% → FAIL (ไม่ใช่ 40/10=400%)
        sizes = {'small': 10, 'medium': 200, 'large': 500}
        ok, _ = validate_output(self._df(40), sizes)
        self.assertFalse(ok)

    def test_single_table_sizes(self):
        ok, label = validate_output(self._df(80), {'only': 100})
        self.assertTrue(ok)
        self.assertEqual(label, 'GATE PASS')


# ═══════════════════════════════════════════════════════════════════════
# 2. BASE TABLE SELECTION — Priority 1 / 2 / 3
# ═══════════════════════════════════════════════════════════════════════

class TestBuildJoinedBaseTable(unittest.TestCase):

    def _run(self, conn):
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        df, base = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        return df, base, fk_pairs

    def test_priority1_exact_match_employees(self):
        _, base, _ = self._run(make_employees_db())
        self.assertEqual(base, 'employees')

    def test_priority2_keyword_match_user_orders(self):
        _, base, _ = self._run(make_user_orders_db())
        self.assertEqual(base, 'user_orders')

    def test_priority3_fallback_largest_skips_geo(self):
        _, base, _ = self._run(make_fallback_geo_db())
        self.assertEqual(base, 'main_data')

    def test_priority3_geo_not_selected(self):
        conn = make_fallback_geo_db()
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        _, base  = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        self.assertFalse(any(skip in base.lower() for skip in SKIP_TABLES),
                         f"Base table '{base}' contains skip keyword")

    def test_no_fk_uses_largest_table(self):
        conn     = make_no_id_db()
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        self.assertEqual(fk_pairs, [])
        df, base = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        self.assertEqual(base, 'alpha')   # 5 rows > 2 rows


# ═══════════════════════════════════════════════════════════════════════
# 3. JOIN RESULT — rows, cols, left join behaviour
# ═══════════════════════════════════════════════════════════════════════

class TestBuildJoinedResult(unittest.TestCase):

    def test_employees_join_has_dept_name(self):
        conn     = make_employees_db()
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        df, _    = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        self.assertIn('dept_name', df.columns)

    def test_employees_join_preserves_all_5_rows(self):
        conn     = make_employees_db()
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        df, _    = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        self.assertEqual(len(df), 5)

    def test_user_orders_join_preserves_all_6_orders(self):
        conn     = make_user_orders_db()
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        df, _    = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        self.assertEqual(len(df), 6)

    def test_user_orders_join_adds_email(self):
        conn     = make_user_orders_db()
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        df, _    = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        self.assertIn('email', df.columns)

    def test_no_fk_returns_base_table_unchanged(self):
        conn     = make_no_id_db()
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        df, _    = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        self.assertEqual(len(df), 5)
        self.assertEqual(set(df.columns), {'name', 'age', 'salary'})

    def test_join_result_cols_gt_base_table_cols(self):
        conn     = make_employees_db()
        tables   = _get_tables(conn)
        fk_pairs = detect_foreign_keys(conn, tables)
        base_cols = len(pd.read_sql_query("SELECT * FROM employees LIMIT 1", conn).columns)
        df, _    = build_joined_dataset(conn, tables, fk_pairs)
        conn.close()
        self.assertGreater(len(df.columns), base_cols)


# ═══════════════════════════════════════════════════════════════════════
# 4. COLUMN RENAME ON DUPLICATE
# ═══════════════════════════════════════════════════════════════════════

class TestColumnRenameOnDuplicate(unittest.TestCase):

    def setUp(self):
        self.conn = make_duplicate_col_db()
        tables       = _get_tables(self.conn)
        fk_pairs     = detect_foreign_keys(self.conn, tables)
        self.df, _   = build_joined_dataset(self.conn, tables, fk_pairs)

    def tearDown(self):
        self.conn.close()

    def test_no_duplicate_columns(self):
        cols = list(self.df.columns)
        self.assertEqual(len(cols), len(set(cols)))

    def test_base_table_category_preserved(self):
        self.assertIn('category', self.df.columns)

    def test_duplicate_column_renamed_with_table_prefix(self):
        self.assertIn('users_category', self.df.columns)

    def test_join_key_appears_exactly_once(self):
        uid_count = list(self.df.columns).count('user_id')
        self.assertEqual(uid_count, 1)

    def test_all_rows_preserved_after_rename(self):
        self.assertEqual(len(self.df), 3)

    def test_renamed_column_has_correct_values(self):
        self.assertTrue(self.df['users_category'].notna().all())


# ═══════════════════════════════════════════════════════════════════════
# 5. FK DETECTION EDGE CASES
# ═══════════════════════════════════════════════════════════════════════

class TestFKEdgeCases(unittest.TestCase):

    def test_overlap_exactly_50pct_not_detected(self):
        # 1 shared / min(2,2) = 0.50 → NOT > 0.5 → not detected
        conn     = make_overlap_50pct_db()
        tables   = _get_tables(conn)
        pairs    = detect_foreign_keys(conn, tables)
        conn.close()
        self.assertEqual(pairs, [], f"50% overlap should not trigger FK, got: {pairs}")

    def test_overlap_above_50pct_detected(self):
        # 2 shared / min(3,3) = 0.667 → detected
        conn     = make_overlap_above_50pct_db()
        tables   = _get_tables(conn)
        pairs    = detect_foreign_keys(conn, tables)
        conn.close()
        self.assertGreater(len(pairs), 0)

    def test_all_detected_overlaps_above_threshold(self):
        conn   = make_employees_db()
        tables = _get_tables(conn)
        pairs  = detect_foreign_keys(conn, tables)
        conn.close()
        for _, _, _, overlap in pairs:
            self.assertGreater(overlap, 0.5,
                               f"Detected FK has overlap {overlap} ≤ 0.5")

    def test_empty_table_does_not_crash(self):
        conn   = make_empty_table_db()
        tables = _get_tables(conn)
        try:
            pairs = detect_foreign_keys(conn, tables)
        except Exception as e:
            self.fail(f"detect_foreign_keys crashed on empty table: {e}")
        conn.close()

    def test_empty_table_yields_no_fk(self):
        # empty_dim has 0 rows → vals2 is empty → skipped → no FK
        conn   = make_empty_table_db()
        tables = _get_tables(conn)
        pairs  = detect_foreign_keys(conn, tables)
        conn.close()
        self.assertEqual(pairs, [])

    def test_no_id_column_yields_no_fk(self):
        conn   = make_no_id_db()
        tables = _get_tables(conn)
        pairs  = detect_foreign_keys(conn, tables)
        conn.close()
        self.assertEqual(pairs, [])

    def test_fk_column_name_contains_id(self):
        conn   = make_employees_db()
        tables = _get_tables(conn)
        pairs  = detect_foreign_keys(conn, tables)
        conn.close()
        for _, _, col, _ in pairs:
            self.assertIn('id', col.lower(),
                          f"FK column '{col}' does not contain 'id'")


# ═══════════════════════════════════════════════════════════════════════
# 6. FORBIDDEN KEYWORD TYPO — _lenght (Olist intentional typo)
# ═══════════════════════════════════════════════════════════════════════

class TestForbiddenKeywordTypo(unittest.TestCase):

    # explicit keywords (ใน FORBIDDEN_TARGET_KEYWORDS)
    def test_product_name_lenght_in_keywords(self):
        self.assertTrue(is_forbidden_target('product_name_lenght'))

    def test_product_description_lenght_in_keywords(self):
        self.assertTrue(is_forbidden_target('product_description_lenght'))

    # suffix rule (ใน FORBIDDEN_TARGET_SUFFIXES)
    def test_title_lenght_caught_by_suffix(self):
        self.assertTrue(is_forbidden_target('title_lenght'))

    def test_desc_lenght_caught_by_suffix(self):
        self.assertTrue(is_forbidden_target('desc_lenght'))

    def test_any_col_ending_lenght_is_forbidden(self):
        for col in ['name_lenght', 'body_lenght', 'text_lenght']:
            with self.subTest(col=col):
                self.assertTrue(is_forbidden_target(col))

    # correct spelling also blocked
    def test_length_correct_spelling_also_forbidden(self):
        self.assertTrue(is_forbidden_target('product_name_length'))
        self.assertTrue(is_forbidden_target('desc_length'))

    # non-forbidden sanity check
    def test_business_columns_not_forbidden(self):
        for col in ['review_score', 'churn', 'order_status', 'price', 'payment_value']:
            with self.subTest(col=col):
                self.assertFalse(is_forbidden_target(col))

    def test_target_detection_skips_lenght_columns(self):
        df = pd.DataFrame({
            'product_name_lenght': [10, 20, 30],
            'review_score':        [1, 3, 5],
        })
        from tests.test_scout_output import detect_target_column
        target = detect_target_column(df)
        self.assertEqual(target, 'review_score')
        self.assertNotEqual(target, 'product_name_lenght')


if __name__ == '__main__':
    unittest.main()
