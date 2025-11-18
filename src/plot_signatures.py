from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_name(name):
    return "".join(c if c.isalnum() else "_" for c in name)[:50]


def main():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    data_path = project_root / "data" / "processed" / "gtd_group_text_subset.csv"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "baseline_tfidf.joblib"
    model_bundle = joblib.load(model_path)

    vectorizer = model_bundle["vectorizer"]
    clf = model_bundle["classifier"]
    label_encoder = model_bundle["label_encoder"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    class_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

    top_n_features = 10

    for class_idx, group_name in enumerate(class_labels):
        coef = clf.coef_[class_idx]
        top_indices = np.argsort(coef)[-top_n_features:][::-1]
        top_features = feature_names[top_indices]
        top_weights = coef[top_indices]

        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_weights)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Weight")
        ax.set_title(group_name, fontsize=10)
        fig.tight_layout()

        fname = figures_dir / f"signature_top10_{safe_name(group_name)}.png"
        fig.savefig(fname, dpi=300)
        plt.close(fig)

    W = clf.coef_.astype(float)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    W_norm = W / norms
    sim_matrix = W_norm @ W_norm.T

    df = pd.read_csv(data_path)
    counts = df["gname"].value_counts()
    top_k = min(15, len(class_labels))
    top_groups = counts.index[:top_k]

    name_to_idx = {name: idx for idx, name in enumerate(class_labels)}
    top_indices = [name_to_idx[g] for g in top_groups if g in name_to_idx]

    sim_top = sim_matrix[np.ix_(top_indices, top_indices)]
    labels_top = np.array([class_labels[i] for i in top_indices])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_top, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels_top)))
    ax.set_yticks(np.arange(len(labels_top)))
    ax.set_xticklabels(labels_top, rotation=90, fontsize=6)
    ax.set_yticklabels(labels_top, fontsize=6)
    ax.set_xlabel("Group")
    ax.set_ylabel("Group")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figures_dir / "signature_similarity_top15.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()

