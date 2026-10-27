# for data manipulation
import pandas as pd
import sklearn
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "week_3_practice/data/playstore_revenue_analysis.csv"

# A small number of rows contain an unescaped embedded quote in app_name,
# which breaks CSV tokenization; skip just those rows rather than failing.
df = pd.read_csv(DATASET_PATH, on_bad_lines="skip")
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['app_id', 'app_name', 'updated_date'], inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
df['app_category'] = label_encoder.fit_transform(df['app_category'])
df['free_or_paid'] = label_encoder.fit_transform(df['free_or_paid'])
df['content_rating'] = label_encoder.fit_transform(df['content_rating'])
df['screentime_category'] = label_encoder.fit_transform(df['screentime_category'])

# Define target variable
target_col = 'adv_revenue'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
