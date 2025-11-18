from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "gtd_group_text_subset.csv"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    models_dir = project_root / "models"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    X = df["text_combined"].astype(str).values
    y = df["gname"].astype(str).values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    clf = LinearSVC(
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train_vec, y_train)

    labels = np.unique(y_encoded)
    target_names = le.inverse_transform(labels)

    y_val_pred = clf.predict(X_val_vec)
    y_test_pred = clf.predict(X_test_vec)

    val_report = classification_report(
        y_val,
        y_val_pred,
        labels=labels,
        target_names=target_names,
    )

    test_report = classification_report(
        y_test,
        y_test_pred,
        labels=labels,
        target_names=target_names,
    )

    with open(reports_dir / "baseline_tfidf_val_report.txt", "w") as f:
        f.write(val_report)

    with open(reports_dir / "baseline_tfidf_test_report.txt", "w") as f:
        f.write(test_report)

    cm = confusion_matrix(
        y_test,
        y_test_pred,
        labels=labels,
    )

    support = np.array([(y_test == lab).sum() for lab in labels])
    top_k = min(15, len(labels))
    top_idx = np.argsort(support)[-top_k:][::-1]

    labels_top = labels[top_idx]
    target_names_top = target_names[top_idx]
    cm_top = cm[np.ix_(top_idx, top_idx)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_top, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels_top)))
    ax.set_yticks(np.arange(len(labels_top)))
    ax.set_xticklabels(target_names_top, rotation=90, fontsize=6)
    ax.set_yticklabels(target_names_top, fontsize=6)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_tfidf_confusion_matrix_top15.png", dpi=300)
    plt.close(fig)

    model_bundle = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "label_encoder": le,
    }

    joblib.dump(
        model_bundle,
        models_dir / "baseline_tfidf.joblib",
    )


if __name__ == "__main__":
    main()

