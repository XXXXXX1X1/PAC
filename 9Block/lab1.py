import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# =========================
# 1. Загрузка данных
# =========================

df = pd.read_csv("/Users/xxx/Desktop/Учеба/Python/Pac/9Block/titanic_prepared.csv")

# Целевая переменная
y = df["label"]

# Матрица признаков: все столбцы, кроме label
X = df.drop("label", axis=1)
# =========================
# 2. Деление на train / test (10% в test)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,  # 10% в тест
    random_state=42,
    stratify=y,  # стратификация по метке
)

print("Размер обучающей выборки (train):", X_train.shape)
print("Размер тестовой выборки (test): ", X_test.shape)


# =========================
# 3. Масштабирование для логистической регрессии
# =========================


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 4. Лаба 9.1 — три модели (с GridSearchCV)
# =========================

# ---------- 4.1. Дерево решений (Decision Tree) с GridSearchCV ----------

# tree_base = DecisionTreeClassifier(random_state=42)
#
# tree_param_grid = {
#     "max_depth": [3, 4, 5, 6, None],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "criterion": ["gini", "entropy"],
# }
#
# tree_grid = GridSearchCV(
#     estimator=tree_base,
#     param_grid=tree_param_grid,
#     scoring="accuracy",  # метрика качества
#     cv=5,                # 5-кратная кросс-валидация
#     n_jobs=-1,           # использовать все ядра
# )
#
# tree_grid.fit(X_train, y_train)
#
# tree_clf = tree_grid.best_estimator_
# y_pred_tree = tree_clf.predict(X_test)
# acc_tree = accuracy_score(y_test, y_pred_tree)


tree_clf = DecisionTreeClassifier(
    max_depth=5,  # разумное ограничение глубины
    random_state=42,
)

tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

print(f"  Точность на тесте: {acc_tree:.4f}")

# print("\n[9.1] Дерево решений (DecisionTreeClassifier)")
# print("  Лучшие параметры (по CV):", tree_grid.best_params_)
# print(f"  Средняя точность на CV:  {tree_grid.best_score_:.4f}")
# print(f"  Точность на тесте:       {acc_tree:.4f}")


# ---------- 4.2. XGBoost с GridSearchCV ----------

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
)

xgb_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

xgb_grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

xgb_grid.fit(X_train, y_train)

xgb_clf = xgb_grid.best_estimator_
y_pred_xgb = xgb_clf.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

print("\n[9.1] XGBoost (XGBClassifier)")
print("  Лучшие параметры (по CV):", xgb_grid.best_params_)
print(f"  Средняя точность на CV:  {xgb_grid.best_score_:.4f}")
print(f"  Точность на тесте:       {acc_xgb:.4f}")


# ---------- 4.3. Логистическая регрессия с GridSearchCV ----------

logreg_base = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
)

logreg_param_grid = {
    "C": [0.1, 0.5, 1.0, 2.0, 5.0],
}

logreg_grid = GridSearchCV(
    estimator=logreg_base,
    param_grid=logreg_param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

logreg_grid.fit(X_train_scaled, y_train)

logreg_clf = logreg_grid.best_estimator_
y_pred_lr = logreg_clf.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("\n[9.1] Логистическая регрессия (LogisticRegression)")
print("  Лучшие параметры (по CV):", logreg_grid.best_params_)
print(f"  Средняя точность на CV:  {logreg_grid.best_score_:.4f}")
print(f"  Точность на тесте:       {acc_lr:.4f}")


# =========================
# 5. Важность признаков и дерево на 2 признаках (9.1)
# =========================

# Используем обученное дерево tree_clf (с лучшими параметрами)
importances = tree_clf.feature_importances_
feature_names = X_train.columns

# Сортируем признаки по убыванию важности
feat_imp = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True,
)

print("\n[9.1] Важность признаков по дереву решений:")
for name, imp in feat_imp:
    print(f"{name:20s}  важность = {imp:.4f}")

# Берём два самых важных признака
top2_features = [name for name, _ in feat_imp[:2]]
print("\n[9.1] Два самых важных признака:", top2_features)

# Обучаем новое дерево только на этих двух признаках
X_train_top2 = X_train[top2_features]
X_test_top2 = X_test[top2_features]

tree_top2 = DecisionTreeClassifier(
    max_depth=4,
    criterion="entropy",
    random_state=42,
)
tree_top2.fit(X_train_top2, y_train)
y_pred_tree_top2 = tree_top2.predict(X_test_top2)
acc_tree_top2 = accuracy_score(y_test, y_pred_tree_top2)

print(f"[9.1] Точность дерева на 2 признаках (тест): {acc_tree_top2:.4f}")


# ============================================================
# 9.2. Реализация собственного случайного леса MyRandomForest
# ============================================================


class MyRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        criterion="gini",
        random_state=None,
    ):
        # Просто сохраняем параметры
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        # trees_ появится в fit

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]
        self.trees_ = []

        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_estimators):
            # Бутстрэп по объектам: случайная выборка с возвращением
            indices = rng.randint(0, n_samples, size=n_samples)
            X_boot = X[indices]
            y_boot = y[indices]

            # Создаём дерево с ограничениями по глубине и числу признаков
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                criterion=self.criterion,
                splitter="best",
                # Чтобы деревья были разными, добавляем i к random_state
                random_state=(
                    None if self.random_state is None else self.random_state + i
                ),
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)

        # Предсказания всех деревьев: (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_])

        n_samples = all_preds.shape[1]  # а не X.shape[1]
        y_pred = []

        for j in range(n_samples):
            # предсказания всех деревьев для j-го объекта
            values, counts = np.unique(all_preds[:, j], return_counts=True)
            y_pred.append(values[np.argmax(counts)])

        return np.array(y_pred)


# =========================
# 9.2. Обучение MyRandomForest и сравнение с одним деревом
# =========================


# Обучаем наш собственный случайный лес

my_rf = MyRandomForest(
    n_estimators=200,   # количество деревьев
    max_depth=None,     # глубокие деревья
    max_features="sqrt",
    criterion="entropy",
    random_state=42,
)
my_rf.fit(X_train, y_train)

y_pred_my_rf = my_rf.predict(X_test)
acc_my_rf = accuracy_score(y_test, y_pred_my_rf)

print("\n[9.2] Собственный случайный лес (MyRandomForest)")
print(f"  Точность леса на тесте:           {acc_my_rf:.4f}")
print(f"  Точность одного дерева на тесте:  {acc_tree:.4f}")




