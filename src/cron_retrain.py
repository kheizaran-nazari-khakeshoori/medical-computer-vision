"""Scheduling nightly retraining with cron."""
# crontab entry: 0 2 * * * /usr/bin/python -m src.train
CRON_EXPR = "0 2 * * *"
def schedule_retrain():
    print(f"scheduled retraining at {CRON_EXPR}")
    return CRON_EXPR
