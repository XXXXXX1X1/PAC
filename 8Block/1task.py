import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== 1. ЗАГРУЗКА И ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ====================

# Загрузка исходных данных (файл лежит рядом со скриптом)
df = pd.read_csv("/home/alex/PycharmProjects/PAC/8Block/wells_info_with_prod.csv")


# ---- 1.1. Оставляем один сырой столбец с датой и один с категорией ----
# Сырой столбец с датой
df["SpudDate"] = pd.to_datetime(df["SpudDate"])
# Сырой категориальный столбец
df["operatorNameIHS"] = df["operatorNameIHS"]

# ---- 1.2. Признаки из даты бурения (SpudDate) ----
df["SpudMonth"] = df["SpudDate"].dt.month      # месяц бурения (1–12)
df["SpudQuarter"] = df["SpudDate"].dt.quarter  # квартал бурения (1–4)
df["SpudYear"] = df["SpudDate"].dt.year        # год бурения

# ---- 1.3. Рабочие даты для интервалов ----
df["PermitDate"] = pd.to_datetime(df["PermitDate"])
df["CompletionDate"] = pd.to_datetime(df["CompletionDate"])
df["FirstProductionDate"] = pd.to_datetime(df["FirstProductionDate"])

# Временные интервалы (в днях)
df["PermitToSpudDays"] = (df["SpudDate"] - df["PermitDate"]).dt.days
df["DrillingDurationDays"] = (df["CompletionDate"] - df["SpudDate"]).dt.days
df["CompletionToProductionDays"] = (
    df["FirstProductionDate"] - df["CompletionDate"]
).dt.days

# ---- 1.4. Технологические признаки ----
# Суммарное количество проппанта и воды
df["TotalProppant"] = df["PROP_PER_FOOT"] * df["LATERAL_LENGTH_BLEND"]
df["TotalWater"] = df["WATER_PER_FOOT"] * df["LATERAL_LENGTH_BLEND"]

# Интенсивность проппанта на фут (может помогать ловить "агрессивные" дизайны)
df["ProppantIntensity"] = df["PROP_PER_FOOT"] / df["LATERAL_LENGTH_BLEND"]

# ---- 1.5. География: расстояние между устьем и забоем (в градусах) ----
df["LateralDistance"] = np.sqrt(
    (df["LatWGS84"] - df["BottomHoleLatitude"]) ** 2
    + (df["LonWGS84"] - df["BottomHoleLongitude"]) ** 2
)

# ---- 1.6. Категоризация длины ствола ----
df["LateralLengthCategory"] = pd.cut(
    df["LATERAL_LENGTH_BLEND"],
    bins=[0, 5000, 8000, float("inf")],
    labels=["short", "medium", "long"],
)

# ==================== 2. ФОРМИРОВАНИЕ X И y ====================

# Целевая переменная строго по условию
y = df["Prod1Year"].astype(float)

# Выбираем признаки, которые пойдут в X
# (здесь и новые, и исходные числовые, и сырой дата/категориальный)
feature_cols = [
    # признаки из даты SpudDate
    "SpudMonth",
    "SpudQuarter",
    "SpudYear",

    # интервалы по времени
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

    # сырые столбцы (по требованию задания)
    "SpudDate",          # сырая дата
    "operatorNameIHS",   # сырая категория

    # опционально — категориальная длина ствола, тоже сырьём
    "LateralLengthCategory",
]

X = df[feature_cols].copy()

# ==================== 3. РАЗБИЕНИЕ НА TRAIN / TEST ====================

# 80% — train, 20% — test; случайное разбиение, как в методичке
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ==================== 4. МАСШТАБИРОВАНИЕ (Только ЧИСЛА) ====================

# Масштабируем только числовые признаки.
# Даты и категориальные столбцы — НЕ трогаем.
num_cols = X_train.select_dtypes(include=[np.number]).columns

scaler_X = StandardScaler()

# fit + transform на train
X_train_scaled_num = scaler_X.fit_transform(X_train[num_cols])
# только transform на test
X_test_scaled_num = scaler_X.transform(X_test[num_cols])

# Копии датафреймов, куда вернём масштабированные значения
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = X_train_scaled_num
X_test_scaled[num_cols] = X_test_scaled_num

# Масштабируем целевую переменную так же: fit на train, transform на test
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# ==================== 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================

# Полный датасет с признаками можно сохранить на всякий
df.to_csv("wells_with_features.csv", index=False)

# Масштабированные данные для обучения моделей
X_train_scaled.to_csv("X_train_scaled.csv", index=False)
X_test_scaled.to_csv("X_test_scaled.csv", index=False)

pd.DataFrame({"Prod1Year_scaled": y_train_scaled.flatten()}).to_csv(
    "y_train_scaled.csv", index=False
)
pd.DataFrame({"Prod1Year_scaled": y_test_scaled.flatten()}).to_csv(
    "y_test_scaled.csv", index=False
)

print("✅ Готово:")
print("  X_train_scaled.csv, X_test_scaled.csv, y_train_scaled.csv, y_test_scaled.csv сохранены.")
print(f"  Числовых признаков под масштабированием: {len(num_cols)}")
print("  Сырой дата-столбец в X: SpudDate")
print("  Сырой категориальный столбец в X: operatorNameIHS")
