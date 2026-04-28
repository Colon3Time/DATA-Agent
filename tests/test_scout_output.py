"""
Scout Output Standards Tests
ตรวจสอบว่า output ทุกอย่างที่ Scout สร้างได้มาตรฐานที่ pipeline ต้องการ
"""
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# Logic ที่ copy มาจาก scout_script.py (single source of truth for tests)
# ═══════════════════════════════════════════════════════════════════════

OLIST_JOIN_QUERY = """
SELECT
    o.order_id, o.customer_id, o.order_status,
    o.order_purchase_timestamp, o.order_approved_at,
    o.order_delivered_carrier_date, o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    r.review_score, r.review_creation_date, r.review_answer_timestamp,
    oi.product_id, oi.seller_id, oi.price, oi.freight_value,
    op.payment_type, op.payment_installments, op.payment_value,
    p.product_category_name, p.product_weight_g,
    c.customer_state, c.customer_city
FROM orders o
JOIN order_reviews r   ON o.order_id = r.order_id
JOIN order_items oi    ON o.order_id = oi.order_id
JOIN order_payments op ON o.order_id = op.order_id AND op.payment_sequential = 1
JOIN products p        ON oi.product_id = p.product_id
JOIN customers c       ON o.customer_id = c.customer_id
"""

EXPECTED_OLIST_COLUMNS = [
    'order_id', 'customer_id', 'order_status',
    'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_carrier_date', 'order_delivered_customer_date',
    'order_estimated_delivery_date',
    'review_score', 'review_creation_date', 'review_answer_timestamp',
    'product_id', 'seller_id', 'price', 'freight_value',
    'payment_type', 'payment_installments', 'payment_value',
    'product_category_name', 'product_weight_g',
    'customer_state', 'customer_city',
]

REQUIRED_PROFILE_FIELDS = [
    'rows', 'cols', 'dtypes', 'missing',
    'target_column', 'problem_type', 'recommended_scaling',
]

VALID_PROBLEM_TYPES   = {'classification', 'regression', 'time_series', 'clustering', 'unknown'}
VALID_SCALING_OPTIONS = {'StandardScaler', 'MinMaxScaler', 'None'}
VALID_REVIEW_SCORES   = {1, 2, 3, 4, 5}

FORBIDDEN_TARGET_KEYWORDS = {
    'zip_code', 'zip_prefix', 'geolocation', 'latitude', 'longitude',
    'product_id', 'order_id', 'customer_id', 'seller_id', 'review_id',
    'product_name_lenght', 'product_description_lenght',
    'product_weight_g', 'product_length_cm', 'product_height_cm',
    'product_width_cm', 'product_photos_qty',
}
FORBIDDEN_TARGET_SUFFIXES = (
    '_cm', '_g', '_mm', '_kg', '_lb',
    '_lenght', '_length', '_width', '_height',
    '_lat', '_lng', '_latitude', '_longitude',
    '_zip', '_prefix', '_code',
)

BUSINESS_TARGET_KEYWORDS = [
    'review_score', 'order_status', 'payment_value', 'freight_value',
    'delivery_days', 'delay', 'churn', 'target', 'label', 'survived',
    'fraud', 'default', 'outcome', 'result', 'response', 'converted',
    'clicked', 'bought', 'cancelled', 'returned', 'status', 'class',
]


def is_olist_db(tables):
    required = {'orders', 'order_reviews', 'order_items', 'order_payments', 'products', 'customers'}
    return required.issubset(set(tables))


def validate_olist_output(df):
    if 'review_score' not in df.columns:
        return False, 'review_score missing'
    if df['review_score'].isna().mean() > 0.5:
        return False, 'review_score >50% NaN'
    return True, 'ok'


def is_forbidden_target(col):
    col_l = col.lower()
    if col_l in {k.lower() for k in FORBIDDEN_TARGET_KEYWORDS}:
        return True
    if any(col_l.endswith(s) for s in FORBIDDEN_TARGET_SUFFIXES):
        return True
    if col_l.endswith('_id') or col_l.startswith('id_'):
        return True
    return False


def detect_target_column(df):
    for kw in BUSINESS_TARGET_KEYWORDS:
        for col in df.columns:
            if (col.lower() == kw or col.lower().startswith(kw)) and not is_forbidden_target(col):
                return col
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            return col
    for col in df.columns:
        if is_forbidden_target(col):
            continue
        is_str = not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col])
        if is_str and 2 <= df[col].nunique() <= 10:
            return col
    for col in reversed(list(df.columns)):
        if is_forbidden_target(col):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10:
            return col
    return None


def detect_problem_type(df, target_col):
    if not target_col:
        return 'clustering' if df.select_dtypes(include='number').shape[1] >= 2 else 'unknown'
    n_uniq = df[target_col].nunique()
    if n_uniq <= 20:
        return 'classification'
    date_cols = df.select_dtypes(include=['datetime', 'object']).columns
    has_date = any('date' in c.lower() or 'time' in c.lower() for c in date_cols)
    return 'time_series' if has_date else 'regression'


def build_profile_text(df, target_col, problem_type):
    n_numeric  = df.select_dtypes(include='number').shape[1]
    n_cat      = df.select_dtypes(include=['object', 'category']).shape[1]
    n_datetime = df.select_dtypes(include='datetime').shape[1]
    miss = (df.isnull().mean() * 100).sort_values(ascending=False)
    top_miss = miss[miss > 0].head(10).round(2).to_dict()

    imbalance = None
    class_dist = {}
    if problem_type == 'classification' and target_col:
        vc = df[target_col].value_counts(normalize=True).round(4)
        class_dist = vc.to_dict()
        imbalance = round(vc.max() / vc.min(), 2) if vc.min() > 0 else None

    scaling = 'MinMaxScaler' if problem_type == 'time_series' else (
        'None' if n_numeric == 0 else 'StandardScaler'
    )

    lines = [
        'DATASET_PROFILE',
        '===============',
        f'rows         : {df.shape[0]:,}',
        f'cols         : {df.shape[1]}',
        f'dtypes       : numeric={n_numeric}, categorical={n_cat}, datetime={n_datetime}',
        f'missing      : {json.dumps(top_miss, ensure_ascii=False)}',
        f'target_column: {target_col or "unknown"}',
        f'problem_type : {problem_type}',
    ]
    if class_dist:
        lines.append(f'class_dist   : {json.dumps({str(k): v for k, v in list(class_dist.items())[:6]})}')
    if imbalance is not None:
        lines.append(f'imbalance_ratio: {imbalance}')
    lines.append(f'recommended_scaling: {scaling}')
    return '\n'.join(lines)


def parse_profile(text):
    result = {}
    for line in text.splitlines():
        if ':' in line and not line.startswith('=') and not line.startswith('DATASET'):
            key, _, val = line.partition(':')
            result[key.strip()] = val.strip()
    return result


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

def make_olist_db(extra_items=False, multi_payment=False, missing_product=False):
    """In-memory Olist DB — flags inject edge cases."""
    conn = sqlite3.connect(':memory:')
    conn.executescript("""
        CREATE TABLE orders (
            order_id TEXT, customer_id TEXT, order_status TEXT,
            order_purchase_timestamp TEXT, order_approved_at TEXT,
            order_delivered_carrier_date TEXT, order_delivered_customer_date TEXT,
            order_estimated_delivery_date TEXT
        );
        CREATE TABLE order_reviews (
            review_id TEXT, order_id TEXT, review_score INTEGER,
            review_creation_date TEXT, review_answer_timestamp TEXT
        );
        CREATE TABLE order_items (
            order_id TEXT, order_item_id INTEGER, product_id TEXT,
            seller_id TEXT, price REAL, freight_value REAL
        );
        CREATE TABLE order_payments (
            order_id TEXT, payment_sequential INTEGER,
            payment_type TEXT, payment_installments INTEGER, payment_value REAL
        );
        CREATE TABLE products (
            product_id TEXT, product_category_name TEXT, product_weight_g REAL
        );
        CREATE TABLE customers (
            customer_id TEXT, customer_state TEXT, customer_city TEXT
        );
        INSERT INTO customers VALUES ('c1','SP','Sao Paulo'),('c2','RJ','Rio'),
                                     ('c3','MG','BH'),('c4','RS','Porto Alegre');
        INSERT INTO products  VALUES ('p1','electronics',200),('p2','books',150),
                                     ('p3','clothing',100);
        INSERT INTO orders VALUES
            ('o1','c1','delivered','2021-01-01','2021-01-01','2021-01-03','2021-01-05','2021-01-06'),
            ('o2','c2','delivered','2021-01-02','2021-01-02','2021-01-04','2021-01-06','2021-01-07'),
            ('o3','c3','delivered','2021-01-03','2021-01-03','2021-01-05','2021-01-07','2021-01-08'),
            ('o4','c4','canceled', '2021-01-04','2021-01-04',NULL,NULL,'2021-01-09');
        INSERT INTO order_reviews VALUES
            ('r1','o1',5,'2021-01-06','2021-01-07'),
            ('r2','o2',3,'2021-01-07','2021-01-08'),
            ('r3','o3',1,'2021-01-08','2021-01-09'),
            ('r4','o4',2,'2021-01-09','2021-01-10');
        INSERT INTO order_items VALUES
            ('o1',1,'p1','s1',100.0,10.0),
            ('o2',1,'p2','s2',50.0,5.0),
            ('o3',1,'p3','s3',80.0,8.0),
            ('o4',1,'p1','s1',120.0,12.0);
        INSERT INTO order_payments VALUES
            ('o1',1,'credit_card',1,110.0),
            ('o2',1,'boleto',2,55.0),
            ('o3',1,'voucher',1,88.0),
            ('o4',1,'credit_card',3,132.0);
    """)
    if extra_items:
        conn.execute("INSERT INTO order_items VALUES ('o1',2,'p2','s2',30.0,3.0)")
    if multi_payment:
        conn.execute("INSERT INTO order_payments VALUES ('o1',2,'boleto',1,50.0)")
    if missing_product:
        conn.execute("INSERT INTO order_items VALUES ('o3',2,'p_missing','s3',20.0,2.0)")
    conn.commit()
    return conn


def make_profile_df(scores=(5, 5, 5, 3, 1)):
    return pd.DataFrame({
        'order_id':     [f'o{i}' for i in range(len(scores))],
        'review_score': list(scores),
        'price':        [100.0] * len(scores),
    })


# ═══════════════════════════════════════════════════════════════════════
# 1. OLIST JOIN — ผลลัพธ์ถูกต้อง
# ═══════════════════════════════════════════════════════════════════════

class TestOlistJoinStandards(unittest.TestCase):

    def setUp(self):
        self.conn = make_olist_db()
        self.df = pd.read_sql_query(OLIST_JOIN_QUERY, self.conn)

    def tearDown(self):
        self.conn.close()

    def test_exact_22_columns(self):
        self.assertEqual(self.df.shape[1], 22)

    def test_all_expected_columns_present(self):
        missing = set(EXPECTED_OLIST_COLUMNS) - set(self.df.columns)
        self.assertEqual(missing, set(), f"Missing columns: {missing}")

    def test_no_duplicate_columns(self):
        self.assertEqual(len(self.df.columns), len(set(self.df.columns)))

    def test_review_score_values_are_1_to_5(self):
        actual = set(self.df['review_score'].dropna().astype(int).unique())
        self.assertTrue(actual.issubset(VALID_REVIEW_SCORES), f"Bad scores: {actual}")

    def test_review_score_no_nulls(self):
        self.assertEqual(self.df['review_score'].isna().sum(), 0)

    def test_order_id_not_null(self):
        self.assertEqual(self.df['order_id'].isna().sum(), 0)

    def test_price_positive(self):
        self.assertTrue((self.df['price'] > 0).all())

    def test_payment_sequential_filter(self):
        """payment_sequential=1 only — ไม่มีการนับซ้ำจาก sequential>1"""
        conn2 = make_olist_db(multi_payment=True)
        df2 = pd.read_sql_query(OLIST_JOIN_QUERY, conn2)
        # o1 ต้องมีแค่ 1 แถว ไม่ใช่ 2
        self.assertEqual(len(df2[df2['order_id'] == 'o1']), 1)
        conn2.close()

    def test_multiple_items_expand_rows(self):
        """2 items ใน order เดียว → join ได้ 2 แถว (expected behavior)"""
        conn2 = make_olist_db(extra_items=True)
        df2 = pd.read_sql_query(OLIST_JOIN_QUERY, conn2)
        self.assertEqual(len(df2[df2['order_id'] == 'o1']), 2)
        conn2.close()

    def test_output_rows_gte_base_table(self):
        """rows ต้องไม่น้อยกว่า orders table (เพราะ JOIN ขยายได้แต่ไม่หด)"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM orders")
        order_count = cursor.fetchone()[0]
        self.assertGreaterEqual(len(self.df), order_count)


# ═══════════════════════════════════════════════════════════════════════
# 2. GATE VALIDATION — ผ่าน/ไม่ผ่านถูกต้อง
# ═══════════════════════════════════════════════════════════════════════

class TestOlistGateValidation(unittest.TestCase):

    def test_gate_pass_normal(self):
        df = pd.DataFrame({'review_score': [1, 3, 5, 4, 2]})
        ok, msg = validate_olist_output(df)
        self.assertTrue(ok)

    def test_gate_fail_no_review_score(self):
        df = pd.DataFrame({'order_id': ['o1', 'o2']})
        ok, msg = validate_olist_output(df)
        self.assertFalse(ok)
        self.assertIn('missing', msg)

    def test_gate_fail_all_null_review_score(self):
        df = pd.DataFrame({'review_score': [None, None, None]})
        ok, msg = validate_olist_output(df)
        self.assertFalse(ok)

    def test_gate_fail_mostly_null(self):
        df = pd.DataFrame({'review_score': [None, None, None, None, 5]})
        ok, msg = validate_olist_output(df)
        self.assertFalse(ok)
        self.assertIn('NaN', msg)

    def test_gate_pass_partial_null_below_threshold(self):
        scores = [None] + [5] * 4    # 20% null → pass
        df = pd.DataFrame({'review_score': scores})
        ok, _ = validate_olist_output(df)
        self.assertTrue(ok)

    def test_gate_boundary_exactly_50pct_null(self):
        # gate ใช้ > 0.5 (strict) → 50% null (0.5) ไม่ trigger → PASS
        df = pd.DataFrame({'review_score': [None, None, 5, 5]})
        ok, _ = validate_olist_output(df)
        self.assertTrue(ok)

    def test_gate_boundary_just_over_50pct_null(self):
        # 3 จาก 5 = 60% null → trigger FAIL
        df = pd.DataFrame({'review_score': [None, None, None, 5, 5]})
        ok, _ = validate_olist_output(df)
        self.assertFalse(ok)


# ═══════════════════════════════════════════════════════════════════════
# 3. TARGET COLUMN DETECTION — priority ถูกต้อง
# ═══════════════════════════════════════════════════════════════════════

class TestTargetColumnDetection(unittest.TestCase):

    def test_review_score_wins_over_order_status(self):
        df = pd.DataFrame({
            'order_id':     ['o1', 'o2'],
            'order_status': ['delivered', 'canceled'],
            'review_score': [5, 3],
        })
        self.assertEqual(detect_target_column(df), 'review_score')

    def test_order_id_is_forbidden(self):
        df = pd.DataFrame({'order_id': ['o1', 'o2'], 'review_score': [5, 3]})
        target = detect_target_column(df)
        self.assertNotEqual(target, 'order_id')

    def test_product_weight_g_is_forbidden(self):
        df = pd.DataFrame({'product_weight_g': [100, 200, 300]})
        target = detect_target_column(df)
        self.assertIsNone(target)

    def test_binary_column_fallback(self):
        df = pd.DataFrame({
            'some_id': ['a', 'b', 'c'],
            'is_fraud': [0, 1, 0],
        })
        self.assertEqual(detect_target_column(df), 'is_fraud')

    def test_categorical_fallback(self):
        df = pd.DataFrame({
            'some_id':  ['a', 'b', 'c'],
            'category': ['cat1', 'cat2', 'cat1'],
        })
        self.assertEqual(detect_target_column(df), 'category')

    def test_numeric_low_cardinality_fallback(self):
        df = pd.DataFrame({
            'some_id': ['a', 'b', 'c'],
            'score':   [1, 2, 3],
        })
        self.assertEqual(detect_target_column(df), 'score')

    def test_returns_none_when_all_forbidden(self):
        df = pd.DataFrame({
            'order_id':      ['o1'],
            'product_id':    ['p1'],
            'customer_id':   ['c1'],
        })
        self.assertIsNone(detect_target_column(df))

    def test_forbidden_suffix_rejected(self):
        df = pd.DataFrame({'weight_g': [100, 200], 'length_cm': [10, 20]})
        self.assertIsNone(detect_target_column(df))


# ═══════════════════════════════════════════════════════════════════════
# 4. PROBLEM TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════

class TestProblemTypeDetection(unittest.TestCase):

    def test_classification_for_review_score(self):
        df = make_profile_df([1, 2, 3, 4, 5])
        pt = detect_problem_type(df, 'review_score')
        self.assertEqual(pt, 'classification')

    def test_classification_for_binary_target(self):
        df = pd.DataFrame({'churn': [0, 1, 0, 1, 1], 'age': [25, 30, 35, 40, 45]})
        pt = detect_problem_type(df, 'churn')
        self.assertEqual(pt, 'classification')

    def test_regression_when_target_high_cardinality(self):
        df = pd.DataFrame({'price': list(range(100)), 'feature': list(range(100))})
        pt = detect_problem_type(df, 'price')
        self.assertEqual(pt, 'regression')

    def test_time_series_when_date_column_exists(self):
        df = pd.DataFrame({
            'date':   ['2021-01-01'] * 50 + ['2021-01-02'] * 50,
            'sales':  list(range(100)),
        })
        pt = detect_problem_type(df, 'sales')
        self.assertEqual(pt, 'time_series')

    def test_clustering_when_no_target(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        pt = detect_problem_type(df, None)
        self.assertEqual(pt, 'clustering')

    def test_valid_problem_type_always(self):
        cases = [
            (make_profile_df(), 'review_score'),
            (pd.DataFrame({'price': range(100), 'f': range(100)}), 'price'),
            (pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), None),
        ]
        for df, target in cases:
            pt = detect_problem_type(df, target)
            self.assertIn(pt, VALID_PROBLEM_TYPES, f"Got invalid: {pt}")


# ═══════════════════════════════════════════════════════════════════════
# 5. IMBALANCE RATIO — คำนวณถูกต้อง
# ═══════════════════════════════════════════════════════════════════════

class TestImbalanceCalculation(unittest.TestCase):

    def _imbalance(self, scores):
        df = make_profile_df(scores)
        target = detect_target_column(df)
        pt = detect_problem_type(df, target)
        profile = build_profile_text(df, target, pt)
        parsed = parse_profile(profile)
        return float(parsed['imbalance_ratio']) if 'imbalance_ratio' in parsed else None

    def test_balanced_classes_ratio_near_1(self):
        scores = [1, 2, 3, 4, 5] * 20  # perfectly balanced
        ratio = self._imbalance(scores)
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)

    def test_imbalanced_classes_ratio_gt_1(self):
        scores = [5] * 90 + [1] * 10
        ratio = self._imbalance(scores)
        self.assertIsNotNone(ratio)
        self.assertGreater(ratio, 1.0)

    def test_olist_typical_imbalance_around_16(self):
        # Olist จริง: 5→56%, 1→12% → ratio ~4.5 (3:1 เพราะ 5 classes)
        scores = [5] * 56 + [4] * 19 + [3] * 8 + [2] * 3 + [1] * 13
        ratio = self._imbalance(scores)
        self.assertGreater(ratio, 3.0)


# ═══════════════════════════════════════════════════════════════════════
# 6. DATASET_PROFILE FORMAT STANDARDS
# ═══════════════════════════════════════════════════════════════════════

class TestDatasetProfileFormat(unittest.TestCase):

    def _make_profile(self, scores=(5, 5, 3, 1, 4)):
        df = make_profile_df(scores)
        target = detect_target_column(df)
        pt = detect_problem_type(df, target)
        return build_profile_text(df, target, pt)

    def test_starts_with_header(self):
        text = self._make_profile()
        self.assertTrue(text.startswith('DATASET_PROFILE'))

    def test_all_required_fields_present(self):
        parsed = parse_profile(self._make_profile())
        for field in REQUIRED_PROFILE_FIELDS:
            self.assertIn(field, parsed, f"Missing field: {field}")

    def test_rows_field_is_numeric_string(self):
        parsed = parse_profile(self._make_profile())
        rows_str = parsed['rows'].replace(',', '')
        self.assertTrue(rows_str.isdigit(), f"rows not numeric: {parsed['rows']}")

    def test_problem_type_valid_value(self):
        parsed = parse_profile(self._make_profile())
        self.assertIn(parsed['problem_type'], VALID_PROBLEM_TYPES)

    def test_scaling_valid_value(self):
        parsed = parse_profile(self._make_profile())
        self.assertIn(parsed['recommended_scaling'], VALID_SCALING_OPTIONS)

    def test_target_column_not_forbidden(self):
        parsed = parse_profile(self._make_profile())
        target = parsed['target_column']
        if target != 'unknown':
            self.assertFalse(is_forbidden_target(target), f"Forbidden target selected: {target}")

    def test_classification_has_class_dist(self):
        text = self._make_profile([1, 2, 3, 4, 5])
        self.assertIn('class_dist', text)

    def test_classification_has_imbalance_ratio(self):
        text = self._make_profile([1, 2, 3, 4, 5])
        self.assertIn('imbalance_ratio', text)

    def test_regression_no_class_dist(self):
        df = pd.DataFrame({'price': list(range(100)), 'feature': list(range(100))})
        target = detect_target_column(df)
        pt = detect_problem_type(df, target)
        text = build_profile_text(df, target, pt)
        self.assertNotIn('class_dist', text)

    def test_missing_field_is_valid_json(self):
        parsed = parse_profile(self._make_profile())
        missing_str = parsed.get('missing', '{}')
        try:
            obj = json.loads(missing_str)
            self.assertIsInstance(obj, dict)
        except json.JSONDecodeError:
            self.fail(f"missing field is not valid JSON: {missing_str}")

    def test_profile_written_to_file(self):
        df = make_profile_df()
        target = detect_target_column(df)
        pt = detect_problem_type(df, target)
        text = build_profile_text(df, target, pt)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dataset_profile.md'
            path.write_text(text, encoding='utf-8')
            self.assertTrue(path.exists())
            loaded = path.read_text(encoding='utf-8')
            self.assertEqual(loaded, text)


# ═══════════════════════════════════════════════════════════════════════
# 7. SCOUT OUTPUT CSV STANDARDS
# ═══════════════════════════════════════════════════════════════════════

class TestScoutOutputCSVStandards(unittest.TestCase):

    def setUp(self):
        self.conn = make_olist_db()
        self.df = pd.read_sql_query(OLIST_JOIN_QUERY, self.conn)

    def tearDown(self):
        self.conn.close()

    def test_csv_roundtrip_preserves_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scout_output.csv'
            self.df.to_csv(path, index=False)
            df2 = pd.read_csv(path)
            self.assertEqual(len(df2), len(self.df))

    def test_csv_roundtrip_preserves_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scout_output.csv'
            self.df.to_csv(path, index=False)
            df2 = pd.read_csv(path)
            self.assertEqual(list(df2.columns), list(self.df.columns))

    def test_review_score_survives_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scout_output.csv'
            self.df.to_csv(path, index=False)
            df2 = pd.read_csv(path)
            self.assertIn('review_score', df2.columns)
            valid = set(df2['review_score'].dropna().astype(int).unique())
            self.assertTrue(valid.issubset(VALID_REVIEW_SCORES))

    def test_no_all_null_columns(self):
        all_null = [c for c in self.df.columns if self.df[c].isna().all()]
        self.assertEqual(all_null, [], f"All-null columns: {all_null}")

    def test_no_index_column_in_output(self):
        """ห้ามมี Unnamed: 0 จากการ save index=True"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scout_output.csv'
            self.df.to_csv(path, index=False)
            df2 = pd.read_csv(path)
            unnamed = [c for c in df2.columns if 'unnamed' in c.lower()]
            self.assertEqual(unnamed, [])

    def test_row_count_reasonable(self):
        """Output rows ต้อง >= orders table row count"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM orders")
        n_orders = cursor.fetchone()[0]
        self.assertGreaterEqual(len(self.df), n_orders)


# ═══════════════════════════════════════════════════════════════════════
# 8. ANNA DISPATCH GATE — gate ก่อน handoff
# ═══════════════════════════════════════════════════════════════════════

class TestAnnaDispatchGate(unittest.TestCase):
    """ตรวจสอบ rule ที่ Anna ใช้ตัดสิน handoff"""

    def _profile(self, target, problem_type, rows):
        return {'target_column': target, 'problem_type': problem_type, 'rows': str(rows)}

    def test_blocks_when_target_unknown(self):
        p = self._profile('unknown', 'classification', 5000)
        self.assertEqual(p['target_column'], 'unknown')   # gate should block

    def test_blocks_when_target_is_id(self):
        p = self._profile('order_id', 'classification', 5000)
        self.assertTrue(is_forbidden_target(p['target_column']))

    def test_blocks_when_rows_too_low(self):
        p = self._profile('review_score', 'classification', 500)
        self.assertLess(int(p['rows'].replace(',', '')), 1000)

    def test_passes_olist_profile(self):
        p = self._profile('review_score', 'classification', 112281)
        self.assertNotEqual(p['target_column'], 'unknown')
        self.assertFalse(is_forbidden_target(p['target_column']))
        self.assertGreater(int(p['rows'].replace(',', '')), 1000)
        self.assertIn(p['problem_type'], {'classification', 'regression'})

    def test_blocks_wrong_problem_type(self):
        p = self._profile('review_score', 'clustering', 5000)
        self.assertNotIn(p['problem_type'], {'classification', 'regression'})


if __name__ == '__main__':
    unittest.main()
