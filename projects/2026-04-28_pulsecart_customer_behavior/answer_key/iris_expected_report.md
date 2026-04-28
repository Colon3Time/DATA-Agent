# Expected Iris Report

Iris should frame this as a customer retention problem for PulseCart, a grocery delivery subscription business.

Minimum expected insight quality:
- The dataset supports predicting `account_status_30d`, a binary churn-risk follow-up label.
- Delivery delay is not merely correlated linearly with churn; risk jumps after a delay threshold around 30 hours.
- Customers with both support tickets and delivery delay are a priority intervention group.
- Discounting has a nonlinear/U-shaped pattern; high discount users are not automatically loyal.
- Product variety below a practical threshold is a churn risk.
- The strongest business action is operational reliability plus targeted retention, not blanket discounts.

A strong final recommendation:
Build a weekly churn-risk workflow. Route high-delay/high-ticket customers to service recovery, offer moderate targeted incentives only after operational issues are fixed, and improve product variety for exposed segments.
