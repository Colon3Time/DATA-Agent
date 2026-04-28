# PulseCart Dataset

Business context: PulseCart is a fictional grocery delivery subscription business.
The dataset contains customer activity, service experience, pricing, product usage, and follow-up status fields.
Use the data to decide what business question is most useful, what column should be modeled if any, and what preparation is required.

Files:
- `pulsecart_raw.csv`: raw customer-level dataset.
- `data_dictionary.md`: schema notes.

Columns:
- `customer_id`: customer key.
- `signup_date`: customer signup date.
- `region`: customer region.
- `plan_type`: subscription plan.
- `acquisition_channel`: acquisition source.
- `age`: customer age.
- `account_tenure_days`: days since signup.
- `avg_order_value`: average basket value.
- `orders_last_90d`: order count in last 90 days.
- `app_sessions_last_30d`: recent app usage.
- `support_tickets_90d`: support tickets in last 90 days.
- `late_delivery_rate`: share of recent deliveries that were late.
- `avg_delivery_delay_hours`: average delay hours.
- `discount_ratio`: discount / gross value.
- `product_variety_score`: score from 1 to 100.
- `return_rate_90d`: returned items / ordered items.
- `competitor_price_index`: competitor price divided by PulseCart price.
- `days_since_last_order`: recency.
- `account_status_30d`: 30-day account follow-up status flag.
- `post_period_refund_flag`: refund flag from the CRM extract.
- `account_note_post_period`: account note from the CRM extract.

Raw shape: 2237 rows x 21 columns.
