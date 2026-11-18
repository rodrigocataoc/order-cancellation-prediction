import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

np.random.seed(42)
sns.set_style("whitegrid")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "orders.csv"
IMG = ROOT / "images"
MODELS = ROOT / "models"
IMG.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
df["order_hour"] = df["order_timestamp"].dt.hour
df["order_weekday"] = df["order_timestamp"].dt.weekday
df["cancelled"] = (df["status"] == "cancelled").astype(int)

FEATURES = [
    "item_count",
    "subtotal",
    "delivery_fee",
    "estimated_delivery_min",
    "order_hour",
    "order_weekday",
]
X = df[FEATURES].copy()
y = df["cancelled"].copy()

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
])
lr_pipe.fit(X_train, y_train)

rf_pipe = Pipeline([
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )),
])
rf_pipe.fit(X_train, y_train)

def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "cm": confusion_matrix(y, pred).tolist(),
        "threshold": float(threshold),
    }, proba

def best_f1_th(y_true, proba):
    prec, rec, ths = precision_recall_curve(y_true, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    idx = int(np.argmax(f1s[:-1]))
    return float(ths[idx]), float(f1s[idx])

print("Dummy test acc", round(accuracy_score(y_test, dummy.predict(X_test)), 4))

lr_v05, lr_proba = evaluate(lr_pipe, X_val, y_val, 0.5)
rf_v05, rf_proba = evaluate(rf_pipe, X_val, y_val, 0.5)
lr_th, _ = best_f1_th(y_val, lr_proba)
rf_th, _ = best_f1_th(y_val, rf_proba)
lr_vbest, _ = evaluate(lr_pipe, X_val, y_val, lr_th)
rf_vbest, _ = evaluate(rf_pipe, X_val, y_val, rf_th)

print("LR th", round(lr_th, 4), "RF th", round(rf_th, 4))
print("LR val best F1", round(lr_vbest["f1"], 4), "CM", lr_vbest["cm"])
print("RF val best F1", round(rf_vbest["f1"], 4), "CM", rf_vbest["cm"])

rf_model = rf_pipe.named_steps["clf"]
imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.round(4).to_dict())

final_pipe = Pipeline([
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )),
])
final_pipe.fit(X_temp, y_temp)

joblib.dump(final_pipe, str(MODELS / "cancellation_rf_pipeline.joblib"))
meta = {
    "model": "RandomForestClassifier",
    "features": FEATURES,
    "threshold": rf_th,
    "threshold_selection": "max F1 on validation set",
    "params": {
        "n_estimators": 300, "max_depth": 5,
        "min_samples_split": 10, "min_samples_leaf": 5,
        "class_weight": "balanced", "random_state": 42,
    },
    "validation_metrics_at_threshold": {k: v for k, v in rf_vbest.items() if k != "cm"},
    "validation_cm": rf_vbest["cm"],
    "note": "Threshold chosen on validation only. Test set was previously inspected in earlier LR experiments.",
}
with open(MODELS / "model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# plots
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts().sort_index()
bars = ax.bar(["Entregue (0)", "Cancelado (1)"], [counts[0], counts[1]], color=["#2ecc71", "#e74c3c"])
ax.set_title("Distribuição das Classes")
ax.set_ylabel("Quantidade")
for bar, v in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 15, f"{v}\n({v/len(y)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(IMG / "class_distribution.png", dpi=150)
plt.close()

def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Entregue", "Cancelado"], yticklabels=["Entregue", "Cancelado"])
    ax.set_title(title); ax.set_ylabel("Real"); ax.set_xlabel("Previsto")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

plot_cm(lr_vbest["cm"], f"Regressão Logística (th={lr_th:.3f}) — Validação", IMG / "confusion_matrix_logistic_regression.png")
plot_cm(rf_vbest["cm"], f"Random Forest (th={rf_th:.3f}) — Validação", IMG / "confusion_matrix_random_forest.png")

prec_lr, rec_lr, _ = precision_recall_curve(y_val, lr_proba)
prec_rf, rec_rf, _ = precision_recall_curve(y_val, rf_proba)
plt.figure(figsize=(7, 5))
plt.plot(rec_lr, prec_lr, label=f'Logistic Regression (PR-AUC={lr_v05["pr_auc"]:.3f})')
plt.plot(rec_rf, prec_rf, label=f'Random Forest (PR-AUC={rf_v05["pr_auc"]:.3f})')
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curva Precision-Recall (Validação)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(IMG / "precision_recall_curve.png", dpi=150); plt.close()

plt.figure(figsize=(7, 4))
imp.plot(kind="barh", color="#3498db")
plt.xlabel("Importância"); plt.title("Importância das Features — Random Forest")
plt.gca().invert_yaxis(); plt.tight_layout()
plt.savefig(IMG / "feature_importance.png", dpi=150); plt.close()

print("Saved images and model to", IMG, MODELS)
