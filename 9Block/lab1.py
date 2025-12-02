import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# =========================
# 1. Загрузка данных
# =========================

df = pd.read_csv("/home/alex/PycharmProjects/PAC/9Block/titanic_prepared.csv")  # поправь путь, если нужно

y = df["label"]
X = df.drop("label", axis=1)


# =========================
# 2. Деление на train / test (10% в test)
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y,
)

print("Размер train:", X_train.shape)
print("Размер test: ", X_test.shape)


# =========================
# 3. Масштабирование для логистической регрессии
# =========================

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 4. Лаба 9.1 — три модели
# =========================

# --- Decision Tree ---
tree_clf = DecisionTreeClassifier(
    max_depth=4,          # контролируем глубину дерева
    criterion="entropy",
    random_state=42,
)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)
print(f"\n[9.1] Decision Tree accuracy:        {acc_tree:.4f}")

# --- XGBoost ---
xgb_clf = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"[9.1] XGBoost accuracy:              {acc_xgb:.4f}")

# --- Logistic Regression ---
logreg_clf = LogisticRegression(
    C=1.0,
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
)
logreg_clf.fit(X_train_scaled, y_train)
y_pred_lr = logreg_clf.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"[9.1] Logistic Regression accuracy:  {acc_lr:.4f}")


# =========================
# 5. Важность признаков и дерево на 2 фичах (9.1)
# =========================

importances = tree_clf.feature_importances_
feature_names = X_train.columns

feat_imp = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True,
)

print("\n[9.1] Важность признаков (Decision Tree):")
for name, imp in feat_imp:
    print(f"{name:20s} importance = {imp:.4f}")

top2_features = [name for name, _ in feat_imp[:2]]
print("\n[9.1] Два самых важных признака:", top2_features)

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
print(f"[9.1] Decision Tree accuracy (2 фичи): {acc_tree_top2:.4f}")


# ============================================================
# 9.2. Реализация собственного случайного леса MyRandomForest
# ============================================================

from sklearn.tree import DecisionTreeClassifier


class MyRandomForest:
    """
    Простейшая реализация случайного леса:

    - bootstrap по объектам: каждое дерево обучается на выборке
      с возвращением из исходных данных;

    - случайный поднабор признаков в каждом узле:
      задаётся через max_features, как в sklearn.DecisionTreeClassifier.
      splitter='best' — выбираем лучший сплит среди выбранных признаков.

    Методы:
    - fit(X, y): обучает n_estimators деревьев
    - predict(X): предсказание по большинству голосов (majority vote)
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        criterion="gini",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.trees: list[DecisionTreeClassifier] = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]
        self.trees = []

        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_estimators):
            # Бутстрэп по объектам
            indices = rng.randint(0, n_samples, size=n_samples)
            X_boot = X[indices]
            y_boot = y[indices]

            # Дерево: случайный поднабор признаков через max_features,
            # сплит выбирается лучший (splitter='best')
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                criterion=self.criterion,
                splitter="best",
                random_state=None if self.random_state is None else self.random_state + i,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = np.asarray(X)
        if not self.trees:
            raise RuntimeError("Сначала нужно вызвать fit().")

        # Предсказания всех деревьев: (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        # Мажоритарное голосование по каждому объекту
        n_samples = X.shape[0]
        y_pred = []

        for j in range(n_samples):
            values, counts = np.unique(all_preds[:, j], return_counts=True)
            y_pred.append(values[np.argmax(counts)])

        return np.array(y_pred)


# =========================
# 9.2. Обучение MyRandomForest и сравнение
# =========================

# Одинокоe дерево: уже обучено выше -> acc_tree

# Наш лес: делаем его мощнее (глубина не ограничена, много деревьев)
my_rf = MyRandomForest(
    n_estimators=200,
    max_depth=None,        # глубокие деревья
    max_features="sqrt",   # классика RandomForest
    criterion="entropy",
    random_state=42,
)
my_rf.fit(X_train, y_train)
y_pred_my_rf = my_rf.predict(X_test)
acc_my_rf = accuracy_score(y_test, y_pred_my_rf)

print(f"\n[9.2] MyRandomForest accuracy:       {acc_my_rf:.4f}")
print(f"[9.2] Обычное Decision Tree:         {acc_tree:.4f}")
