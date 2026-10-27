import pandas as pd

RAW_PATH = "week_3_practice/data/playstore_revenue_analysis.csv"

# Load the raw dataset. A small number of rows contain an app name with an
# unescaped embedded quote (e.g. Retro Arcade "Classic" Games), which breaks
# CSV tokenization. on_bad_lines="skip" drops just those malformed rows
# instead of failing the whole pipeline; the count skipped is reported below.
rows_before = sum(1 for _ in open(RAW_PATH, encoding="utf-8", errors="replace")) - 1
df = pd.read_csv(RAW_PATH, on_bad_lines="skip")
rows_skipped = rows_before - df.shape[0]
if rows_skipped:
    print(f"Skipped {rows_skipped} malformed row(s) while parsing {RAW_PATH}.")

# Validate that the expected columns are present before registering it
expected_columns = [
    "app_id", "app_name", "app_category", "free_or_paid", "content_rating",
    "screentime_category", "app_size_in_mb", "price_in_usd", "number_of_installs",
    "average_screen_time", "active_users", "no_of_short_ads_per_hour",
    "no_of_long_ads_per_hour", "updated_date", "adv_revenue",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Target column preview (adv_revenue):")
print(df["adv_revenue"].describe())
