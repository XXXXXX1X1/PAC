import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# =========================
# 1. Загрузка и подготовка данных
# =========================

# Загрузка Kaggle train.csv
df = pd.read_csv("/home/alex/PycharmProjects/PAC/9Block/train.csv")

# Целевая переменная
y = df["Survived"]

# Уберём лишние столбцы: id, имя, билет, кабина
X_raw = df.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)


def prepare_num(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Кодирование категориальных признаков так же, как в конспекте:
      - Sex -> get_dummies
      - Embarked -> get_dummies с префиксом Emb_
      - Pclass -> get_dummies с префиксом Pclass_
    Остальные числовые оставляем как есть.
    """
    df_num = df_in.drop(["Sex", "Embarked", "Pclass"], axis=1)

    df_sex = pd.get_dummies(df_in["Sex"])                 # female, male
    df_emb = pd.get_dummies(df_in["Embarked"], prefix="Emb")   # Emb_C, Emb_Q, Emb_S
    df_pcl = pd.get_dummies(df_in["Pclass"], prefix="Pclass")  # Pclass_1,2,3

    df_num = pd.concat([df_num, df_sex, df_emb, df_pcl], axis=1)
    return df_num


# Числовая матрица признаков
X_full = prepare_num(X_raw)

# Заполнение пропусков медианой по столбцам
X_full = X_full.fillna(X_full.median(numeric_only=True))

# =========================
# 2. Деление на train / val / test
# =========================
# Стратифицированно по Survived, итог: 60% train, 20% val, 20% test

# Сначала отделяем тест 20%
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_full,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Теперь от оставшихся 80% отделим валидацию 25% (0.8 * 0.25 = 0.2)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val,
)

# Для деревьев и XGB используем необработанные X_train / X_val / X_test
# Для линейных моделей и KNN делаем масштабирование

scaler = MinMaxScaler()
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
# 3. Подбор гиперпараметров
# =========================

# --- 3.1 RandomForest ---
best_rf_acc = 0.0
best_rf_params = None

for n_estimators in [100, 200]:
    for max_depth in [3, 5, 7, None]:
        for min_samples_split in [2, 5]:
            for min_samples_leaf in [1, 2]:
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion="gini",
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)
                y_val_pred = rf.predict(X_val)
                acc = accuracy_score(y_val, y_val_pred)

                if acc > best_rf_acc:
                    best_rf_acc = acc
                    best_rf_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "criterion": "gini",
                    }

print("Лучшие параметры RandomForest:", best_rf_params)
print("Accuracy на валидации (RF):", best_rf_acc)


# --- 3.2 XGBoost ---
best_xgb_acc = 0.0
best_xgb_params = None

for n_estimators in [100, 200]:
    for max_depth in [3, 4, 5]:
        for learning_rate in [0.05, 0.1, 0.2]:
            xgb = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )

            xgb.fit(X_train, y_train)
            y_val_pred = xgb.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)

            if acc > best_xgb_acc:
                best_xgb_acc = acc
                best_xgb_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                }

print("Лучшие параметры XGBoost:", best_xgb_params)
print("Accuracy на валидации (XGB):", best_xgb_acc)


# --- 3.3 Logistic Regression (на масштабированных данных) ---
best_lr_acc = 0.0
best_lr_params = None

for C in [0.01, 0.1, 1.0, 10.0]:
    lr = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
    )
    lr.fit(X_train_scaled, y_train)
    y_val_pred = lr.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_val_pred)

    if acc > best_lr_acc:
        best_lr_acc = acc
        best_lr_params = {"C": C, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000}

print("Лучшие параметры LogisticRegression:", best_lr_params)
print("Accuracy на валидации (LR):", best_lr_acc)


# --- 3.4 KNN (на масштабированных данных) ---
best_knn_acc = 0.0
best_knn_params = None

for n_neighbors in [3, 5, 7, 9]:
    for weights in ["uniform", "distance"]:
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
        )
        knn.fit(X_train_scaled, y_train)
        y_val_pred = knn.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_val_pred)

        if acc > best_knn_acc:
            best_knn_acc = acc
            best_knn_params = {
                "n_neighbors": n_neighbors,
                "weights": weights,
            }

print("Лучшие параметры KNN:", best_knn_params)
print("Accuracy на валидации (KNN):", best_knn_acc)


# =========================
# 4. Обучение финальных моделей на train+val и оценка на test
# =========================

# Объединяем train и val
X_train_val_full = pd.concat([X_train, X_val], axis=0)
y_train_val_full = pd.concat([y_train, y_val], axis=0)

# Для деревьев и XGB используем необработанные данные
X_test_full = X_test.copy()

# RandomForest с лучшими параметрами
rf_best = RandomForestClassifier(
    **best_rf_params,
    random_state=42,
    n_jobs=-1,
)
rf_best.fit(X_train_val_full, y_train_val_full)
y_test_pred_rf = rf_best.predict(X_test_full)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)

# XGBoost с лучшими параметрами
xgb_best = XGBClassifier(
    **best_xgb_params,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
xgb_best.fit(X_train_val_full, y_train_val_full)
y_test_pred_xgb = xgb_best.predict(X_test_full)
test_acc_xgb = accuracy_score(y_test, y_test_pred_xgb)

# Для LR и KNN пересоздаём скейлер на train+val
scaler_full = MinMaxScaler()
X_train_val_scaled_full = pd.DataFrame(
    scaler_full.fit_transform(X_train_val_full),
    columns=X_train_val_full.columns,
    index=X_train_val_full.index,
)
X_test_scaled_full = pd.DataFrame(
    scaler_full.transform(X_test_full),
    columns=X_test_full.columns,
    index=X_test_full.index,
)

# Logistic Regression
lr_best = LogisticRegression(**best_lr_params)
lr_best.fit(X_train_val_scaled_full, y_train_val_full)
y_test_pred_lr = lr_best.predict(X_test_scaled_full)
test_acc_lr = accuracy_score(y_test, y_test_pred_lr)

# KNN
knn_best = KNeighborsClassifier(**best_knn_params)
knn_best.fit(X_train_val_scaled_full, y_train_val_full)
y_test_pred_knn = knn_best.predict(X_test_scaled_full)
test_acc_knn = accuracy_score(y_test, y_test_pred_knn)

print("\n=== Итоговая точность на тестовой части (все признаки) ===")
print(f"RandomForest:      {test_acc_rf:.4f}")
print(f"XGBoost:           {test_acc_xgb:.4f}")
print(f"LogisticRegression:{test_acc_lr:.4f}")
print(f"KNN:               {test_acc_knn:.4f}")


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
    print(f"{name:15s}  importance={imp:.4f}")

top2 = [name for name, _ in feat_imp[:2]]
top4 = [name for name, _ in feat_imp[:4]]
top8 = [name for name, _ in feat_imp[:8]]


def eval_on_features(selected_features):
    """
    Обучаем все 4 модели только на выбранных признаках
    (на train+val) и замеряем accuracy на test.
    """
    print("\n====================================")
    print("Признаки:", selected_features)

    # Подмножество данных
    X_train_val_sub = X_train_val_full[selected_features]
    X_test_sub = X_test_full[selected_features]

    # --- RandomForest ---
    rf = RandomForestClassifier(
        **best_rf_params,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_val_sub, y_train_val_full)
    acc_rf = accuracy_score(y_test, rf.predict(X_test_sub))

    # --- XGBoost ---
    xgb = XGBClassifier(
        **best_xgb_params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train_val_sub, y_train_val_full)
    acc_xgb = accuracy_score(y_test, xgb.predict(X_test_sub))

    # Для LR и KNN снова масштабируем только выбранные признаки
    scaler_sub = MinMaxScaler()
    X_train_val_sub_scaled = scaler_sub.fit_transform(X_train_val_sub)
    X_test_sub_scaled = scaler_sub.transform(X_test_sub)

    # --- Logistic Regression ---
    lr = LogisticRegression(**best_lr_params)
    lr.fit(X_train_val_sub_scaled, y_train_val_full)
    acc_lr = accuracy_score(y_test, lr.predict(X_test_sub_scaled))

    # --- KNN ---
    knn = KNeighborsClassifier(**best_knn_params)
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
