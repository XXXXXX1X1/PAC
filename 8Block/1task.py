import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== 0. СОЗДАЁМ ПАПКУ data ====================

os.makedirs("data", exist_ok=True)   # папка data в 8Block

# ==================== 1. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ====================

df = pd.read_csv("wells_info_with_prod.csv")

df["SpudDate"] = pd.to_datetime(df["SpudDate"])

# --- 1.2. Признаки из даты бурения (SpudDate) ---
df["SpudMonth"] = df["SpudDate"].dt.month          # месяц бурения (1–12)
df["SpudQuarter"] = df["SpudDate"].dt.quarter      # квартал бурения (1–4)
df["SpudYear"] = df["SpudDate"].dt.year            # год бурения скважины

# --- 1.3. Рабочие даты и интервалы ---
df["PermitDate"] = pd.to_datetime(df["PermitDate"])              # дата получения разрешения (permit)
df["CompletionDate"] = pd.to_datetime(df["CompletionDate"])      # дата завершения строительства/заканчивание
df["FirstProductionDate"] = pd.to_datetime(df["FirstProductionDate"])  # дата начала добычи

df["PermitToSpudDays"] = (df["SpudDate"] - df["PermitDate"]).dt.days
# количество дней от разрешения до начала бурения

df["DrillingDurationDays"] = (df["CompletionDate"] - df["SpudDate"]).dt.days
# длительность бурения/строительства скважины в днях

df["CompletionToProductionDays"] = (
    df["FirstProductionDate"] - df["CompletionDate"]
).dt.days
# количество дней от завершения работ до начала добычи

# --- 1.4. Технологические признаки ---
df["TotalProppant"] = df["PROP_PER_FOOT"] * df["LATERAL_LENGTH_BLEND"]
# суммарный объём проппанта по всей длине ствола

df["TotalWater"] = df["WATER_PER_FOOT"] * df["LATERAL_LENGTH_BLEND"]
# суммарный объём закачанной воды по всей длине ствола

df["ProppantIntensity"] = df["PROP_PER_FOOT"] / df["LATERAL_LENGTH_BLEND"]
# "интенсивность" проппанта: проппанта на фут относительно длины (нормировка)

# --- 1.5. География ---
df["LateralDistance"] = np.sqrt(
    (df["LatWGS84"] - df["BottomHoleLatitude"]) ** 2
    + (df["LonWGS84"] - df["BottomHoleLongitude"]) ** 2
)
# горизонтальное расстояние между устьем и забоем по координатам (евклидова дистанция)




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

    # категориальные
    "operatorNameIHS",
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

# список имён только числовых признаков
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

# масштабированный y
pd.DataFrame({"Prod1Year_scaled": y_train_scaled.flatten()}).to_csv(
    "data/y_train_scaled.csv", index=False
)
pd.DataFrame({"Prod1Year_scaled": y_test_scaled.flatten()}).to_csv(
    "data/y_test_scaled.csv", index=False
)

# print(X_train_scaled.shape)
# print(X_test_scaled.shape)

print("\nФайлы сохранены в папку 8Block/data")
