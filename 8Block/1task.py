import pandas as pd
import numpy as np


date_cols = ["PermitDate", "SpudDate", "CompletionDate", "FirstProductionDate"]

df = pd.read_csv(
    "wells_info_with_prod.csv",
    parse_dates=date_cols
)

# --- 2. Целевая переменная ---

y = df["Prod1Year"].copy()

# API — чистый идентификатор, в признаках не нужен
cols_to_drop = ["Prod1Year", "ProdAll", "API"]  # ProdAll убираем как утечку
for c in cols_to_drop:
    if c in df.columns:
        df = df.drop(columns=c)

# --- 3. Признаки из дат ---

# Оставляем одну настоящую дату как есть
keep_date_col = "SpudDate"

# Интервалы между основными этапами
df["days_permit_to_spud"] = (df["SpudDate"] - df["PermitDate"]).dt.days
df["days_spud_to_completion"] = (df["CompletionDate"] - df["SpudDate"]).dt.days
df["days_completion_to_firstprod"] = (
    df["FirstProductionDate"] - df["CompletionDate"]
).dt.days

# Календарные фичи по дате бурения и первой добычи
df["Spud_year"] = df["SpudDate"].dt.year
df["Spud_month"] = df["SpudDate"].dt.month
df["FirstProd_year"] = df["FirstProductionDate"].dt.year
df["FirstProd_month"] = df["FirstProductionDate"].dt.month

# Другие даты больше как явные datetime не нужны
drop_dates = ["PermitDate", "CompletionDate", "FirstProductionDate"]
drop_dates = [c for c in drop_dates if c != keep_date_col]
df = df.drop(columns=drop_dates)

# --- 4. Категориальные признаки ---

cat_cols = ["operatorNameIHS", "formation", "BasinName", "StateName", "CountyName"]
cat_cols = [c for c in cat_cols if c in df.columns]

# Оставляем ОДНУ категориальную как есть
keep_cat_col = "formation"

# Остальные заодно переведём в category
for c in cat_cols:
    df[c] = df[c].astype("category")

cat_to_encode = [c for c in cat_cols if c != keep_cat_col]

# one-hot для остальных категориальных
df = pd.get_dummies(
    df,
    columns=cat_to_encode,
    dummy_na=True  # отдельная категория для NaN
)

# --- 5. Обработка числовых пропусков ---

num_cols = df.select_dtypes(include=["number"]).columns

for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# На этом этапе:
#  - SpudDate — остался как datetime64
#  - formation — остался категориальным/строковым
#  - остальные категориальные ушли в one-hot
#  - числовые фичи + новые интервалы и календарные признаки

X = df.copy()
import pandas as pd

pd.set_option("display.max_columns", None)   # показывать все столбцы
pd.set_option("display.width", 200)         # ширина строки в символах (можно больше)

print(X.head())



print("Форма X:", X.shape)
print("Форма y:", y.shape)
print(X.dtypes.head())
