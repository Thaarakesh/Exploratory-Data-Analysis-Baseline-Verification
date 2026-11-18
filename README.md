# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = sns.load_dataset("titanic")

# 3. Display Dataset
print(df.head())
print(df.shape)
print(df.info())

# 4. Descriptive Statistics
print(df.describe(include="all"))

# 5. Check Missing Values
print(df.isnull().sum())

# 6. Missing Value Percentage
missing = (df.isnull().sum() / len(df)) * 100
print(missing)

# 7. Missing Value Imputation
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns

for col in categorical_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after imputation:")
print(df.isnull().sum())

# 8. Outlier Detection using IQR
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) | (df[col] > upper)
    ]

    print(col, "Outliers:", len(outliers))

# 9. Histogram
df[numeric_cols].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# 10. Boxplot
for col in numeric_cols:
    sns.boxplot(x=df[col])
    plt.title("Outlier Detection: " + col)
    plt.show()

# 11. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# 12. Save Cleaned Dataset
df.to_csv("titanic_cleaned.csv", index=False)

# 13. Final Verification
print("Final Dataset Shape:", df.shape)
print("Total Missing Values:", df.isnull().sum().sum())
print("EDA Completed Successfully")
