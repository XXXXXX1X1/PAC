import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== 0. СОЗДАЁМ ПАПКУ data ====================

os.makedirs("data", exist_ok=True)   # папка data в 8Block

# ==================== 1. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ====================

df = pd.read_csv("wells_info_with_prod.csv")  # раз запускаешь из 8Block, путь можно укоротить

# --- 1.1. Сохраняем исходные столбцы (дата и категориальный) ---
df["SpudDate"] = pd.to_datetime(df["SpudDate"])      # дата (останется в df)
df["operatorNameIHS"] = df["operatorNameIHS"]        # категориальный (останется в df)

# --- 1.2. Признаки из даты бурения (SpudDate) ---
df["SpudMonth"] = df["SpudDate"].dt.month
df["SpudQuarter"] = df["SpudDate"].dt.quarter
df["SpudYear"] = df["SpudDate"].dt.year

# --- 1.3. Рабочие даты и интервалы ---
df["PermitDate"] = pd.to_datetime(df["PermitDate"])
df["CompletionDate"] = pd.to_datetime(df["CompletionDate"])
df["FirstProductionDate"] = pd.to_datetime(df["FirstProductionDate"])

df["PermitToSpudDays"] = (df["SpudDate"] - df["PermitDate"]).dt.days
df["DrillingDurationDays"] = (df["CompletionDate"] - df["SpudDate"]).dt.days
df["CompletionToProductionDays"] = (
    df["FirstProductionDate"] - df["CompletionDate"]
).dt.days

# --- 1.4. Технологические признаки ---
df["TotalProppant"] = df["PROP_PER_FOOT"] * df["LATERAL_LENGTH_BLEND"]
df["TotalWater"] = df["WATER_PER_FOOT"] * df["LATERAL_LENGTH_BLEND"]
df["ProppantIntensity"] = df["PROP_PER_FOOT"] / df["LATERAL_LENGTH_BLEND"]

# --- 1.5. География ---
df["LateralDistance"] = np.sqrt(
    (df["LatWGS84"] - df["BottomHoleLatitude"]) ** 2
    + (df["LonWGS84"] - df["BottomHoleLongitude"]) ** 2
)

# --- 1.6. Категория длины ствола ---
df["LateralLengthCategory"] = pd.cut(
    df["LATERAL_LENGTH_BLEND"],
    bins=[0, 5000, 8000, float("inf")],
    labels=["short", "medium", "long"],
)

# ==================== 2. ФОРМИРОВАНИЕ X И y ====================

y = df["Prod1Year"].astype(float)

feature_cols = [
    # временные
    "SpudMonth",
    "SpudQuarter",
    "SpudYear",

    # интервалы
    "PermitToSpudDays",
    "DrillingDurationDays",
    "CompletionToProductionDays",

    # исходные числовые
    "LATERAL_LENGTH_BLEND",
    "PROP_PER_FOOT",
    "WATER_PER_FOOT",

    # технологические
    "TotalProppant",
    "TotalWater",
    "ProppantIntensity",

    # географический
    "LateralDistance",

    # категориальные (оставляем как есть, без one-hot)
    "operatorNameIHS",
    "LateralLengthCategory",
    "BasinName",
    "formation",
]

X = df[feature_cols].copy()

# ==================== 3. РАЗБИЕНИЕ НА TRAIN / TEST ====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ==================== 4. МАСШТАБИРОВАНИЕ (ТОЛЬКО ЧИСЛОВЫЕ ПРИЗНАКИ) ====================

num_cols = X_train.select_dtypes(include=[np.number]).columns

scaler_X = StandardScaler()

# fit + transform на train
X_train_scaled_num = scaler_X.fit_transform(X_train[num_cols])
# transform на test
X_test_scaled_num = scaler_X.transform(X_test[num_cols])

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = X_train_scaled_num
X_test_scaled[num_cols] = X_test_scaled_num

# Масштабируем целевую переменную
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# ==================== 5. СОХРАНЕНИЕ ====================

# полный df с извлечёнными признаками
df.to_csv("data/wells_with_features.csv", index=False)

# масштабированные X
X_train_scaled.to_csv("data/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/X_test_scaled.csv", index=False)

# масштабированный y (опционально, но полезно)
pd.DataFrame({"Prod1Year_scaled": y_train_scaled.flatten()}).to_csv(
    "data/y_train_scaled.csv", index=False
)
pd.DataFrame({"Prod1Year_scaled": y_test_scaled.flatten()}).to_csv(
    "data/y_test_scaled.csv", index=False
)

print(X_train_scaled.shape)
print(X_test_scaled.shape)

print("\nФайлы сохранены в папку 8Block/data")
