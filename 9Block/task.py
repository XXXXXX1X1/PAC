import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# =========================
# 1. Загрузка и подготовка данных
# =========================

# Загрузка  train.csv
df = pd.read_csv("/Users/xxx/Desktop/Учеба/Python/Pac/9Block/train.csv")

# Целевая переменная
y = df["Survived"]



# Размер семьи: пассажир + родственники на борту
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Признак: один ли человек (без семьи)
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Стоимость на человека
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

# Уберём лишние столбцы: id, имя, билет, кабина
X_raw = df.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)

X_full = X_raw.copy()
# print( df["Sex"].unique())
# print(df["Embarked"].unique())

# Кодируем пол: female -> 0, male -> 1
X_full["Sex"] = X_full["Sex"].map({"female": 0, "male": 1})

# Кодируем порт посадки: C, Q, S -> 0, 1, 2
X_full["Embarked"] = X_full["Embarked"].map({"C": 0, "Q": 1, "S": 2})

# Заполнение пропусков медианой по числовым столбцам
X_full = X_full.fillna(X_full.median(numeric_only=True))


# =========================
# 2. Деление на train / val / test
# =========================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index,
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)


# =========================
# 3. Подбор гиперпараметров через GridSearchCV
# =========================

# --- 3.1 RandomForest ---
rf_param_grid = {
    "n_estimators": [100, 200], # количество деревьев
    "max_depth": [3, 5, 7, None], # максимальная глубина дерева
    "min_samples_split": [2, 5], # минимальное число объектов в узле
    "min_samples_leaf": [1, 2], # минимальное число объектов в листе
}

rf_base = RandomForestClassifier(
    criterion="gini",
    random_state=42,
    n_jobs=-1,
)

rf_grid = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
rf_val_acc = accuracy_score(y_val, best_rf.predict(X_val))

print("Лучшие параметры RandomForest:", rf_grid.best_params_)
print("Accuracy на валидации (RF):", rf_val_acc)


# --- 3.2 XGBoost ---
xgb_param_grid = {
    "n_estimators": [100, 200], # количество деревьев
    "max_depth": [3, 4, 5], # максимальная глубина
    "learning_rate": [0.05, 0.1, 0.2], # Шаг обучения
}

xgb_base = XGBClassifier(
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

xgb_grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_
xgb_val_acc = accuracy_score(y_val, best_xgb.predict(X_val))

print("Лучшие параметры XGBoost:", xgb_grid.best_params_)
print("Accuracy на валидации (XGB):", xgb_val_acc)


# --- 3.3 Logistic Regression (на масштабированных данных) ---
lr_param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
}

lr_base = LogisticRegression(
    max_iter=1000,
)

lr_grid = GridSearchCV(
    estimator=lr_base,
    param_grid=lr_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

lr_grid.fit(X_train_scaled, y_train)

best_lr = lr_grid.best_estimator_
lr_val_acc = accuracy_score(y_val, best_lr.predict(X_val_scaled))

print("Лучшие параметры LogisticRegression:", lr_grid.best_params_)
print("Accuracy на валидации (LR):", lr_val_acc)


# --- 3.4 KNN (на масштабированных данных) ---
knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9], # Сколько соседей учитывать при классификации
    "weights": ["uniform", "distance"], # "uniform" — все соседи равны (каждый голос = 1); "distance" — ближние соседи имеют больший вес, дальние — меньший.

}

knn_base = KNeighborsClassifier()

knn_grid = GridSearchCV(
    estimator=knn_base,
    param_grid=knn_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

knn_grid.fit(X_train_scaled, y_train)

best_knn = knn_grid.best_estimator_
knn_val_acc = accuracy_score(y_val, best_knn.predict(X_val_scaled))

print("Лучшие параметры KNN:", knn_grid.best_params_)
print("Accuracy на валидации (KNN):", knn_val_acc)


# =========================
# 4. Обучение финальных моделей на train+val и оценка на test
# =========================

# Объединяем train и val
X_train_val_full = pd.concat([X_train, X_val], axis=0)
y_train_val_full = pd.concat([y_train, y_val], axis=0)


# --- RandomForest с лучшими параметрами ---
rf_best = RandomForestClassifier(
    **rf_grid.best_params_,
    random_state=42,
    n_jobs=-1,
)
rf_best.fit(X_train_val_full, y_train_val_full)
y_test_pred_rf = rf_best.predict(X_test)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)

# --- XGBoost с лучшими параметрами ---
xgb_best = XGBClassifier(
    **xgb_grid.best_params_,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
xgb_best.fit(X_train_val_full, y_train_val_full)
y_test_pred_xgb = xgb_best.predict(X_test)
test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)

# --- Для LR и KNN пересоздаём скейлер на train+val ---
scaler_full = StandardScaler()
X_train_val_scaled_full = pd.DataFrame(
    scaler_full.fit_transform(X_train_val_full),
    columns=X_train_val_full.columns,
    index=X_train_val_full.index,
)
X_test_scaled_full = pd.DataFrame(
    scaler_full.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

# Logistic Regression
lr_best_final = LogisticRegression(**lr_grid.best_params_)
lr_best_final.fit(X_train_val_scaled_full, y_train_val_full)
y_test_pred_lr = lr_best_final.predict(X_test_scaled_full)
test_acc_lr = accuracy_score(y_test, y_test_pred_lr)

# KNN
knn_best_final = KNeighborsClassifier(**knn_grid.best_params_)
knn_best_final.fit(X_train_val_scaled_full, y_train_val_full)
y_test_pred_knn = knn_best_final.predict(X_test_scaled_full)
test_acc_knn = accuracy_score(y_test, y_test_pred_knn)

print("\n=== Итоговая точность на тестовой части (все признаки) ===")
print(f"RandomForest:       {test_acc_rf:.4f}")
print(f"XGBoost:            {test_acc_xgb:.4f}")
print(f"LogisticRegression: {test_acc_lr:.4f}")
print(f"KNN:                {test_acc_knn:.4f}")


# =========================
# 5. Отбор 2, 4, 8 самых важных признаков с помощью RandomForest
# =========================

importances = rf_best.feature_importances_
feature_names = X_train_val_full.columns

feat_imp = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True,
)

print("\nТоп признаков по важности (RandomForest):")
for name, imp in feat_imp:
    print(f"{name:20s}  importance={imp:.4f}")

top2 = [name for name, _ in feat_imp[:2]]
top4 = [name for name, _ in feat_imp[:4]]
top8 = [name for name, _ in feat_imp[:8]]


def eval_on_features(selected_features):
    """Переобучаем все 4 модели только на выбранных признаках"""
    print("\n====================================")
    print("Признаки:", selected_features)

    # Подмножество данных
    X_train_val_sub = X_train_val_full[selected_features]
    X_test_sub = X_test[selected_features]

    # --- RandomForest ---
    rf = RandomForestClassifier(
        **rf_grid.best_params_,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_val_sub, y_train_val_full)
    acc_rf = accuracy_score(y_test, rf.predict(X_test_sub))

    # --- XGBoost ---
    xgb = XGBClassifier(
        **xgb_grid.best_params_,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train_val_sub, y_train_val_full)
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test_sub))

    # Для LR и KNN снова масштабируем только выбранные признаки
    scaler_sub = StandardScaler()
    X_train_val_sub_scaled = scaler_sub.fit_transform(X_train_val_sub)
    X_test_sub_scaled = scaler_sub.transform(X_test_sub)

    # --- Logistic Regression ---
    lr = LogisticRegression(**lr_grid.best_params_)
    lr.fit(X_train_val_sub_scaled, y_train_val_full)
    acc_lr = accuracy_score(y_test, lr.predict(X_test_sub_scaled))

    # --- KNN ---
    knn = KNeighborsClassifier(**knn_grid.best_params_)
    knn.fit(X_train_val_sub_scaled, y_train_val_full)
    acc_knn = accuracy_score(y_test, knn.predict(X_test_sub_scaled))

    print(f"RandomForest:       {acc_rf:.4f}")
    print(f"XGBoost:            {acc_xgb:.4f}")
    print(f"LogisticRegression: {acc_lr:.4f}")
    print(f"KNN:                {acc_knn:.4f}")


# Оценка на 2 / 4 / 8 признаках
eval_on_features(top2)
eval_on_features(top4)
eval_on_features(top8)
