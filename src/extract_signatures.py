from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "baseline_tfidf.joblib"
    model_bundle = joblib.load(model_path)

    vectorizer = model_bundle["vectorizer"]
    clf = model_bundle["classifier"]
    label_encoder = model_bundle["label_encoder"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    class_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

    records = []
    top_n = 30

    for class_idx, group_name in enumerate(class_labels):
        coef = clf.coef_[class_idx]
        top_indices = np.argsort(coef)[-top_n:][::-1]
        top_features = feature_names[top_indices]
        top_weights = coef[top_indices]

        for feature, weight in zip(top_features, top_weights):
            records.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "weight": float(weight),
                }
            )

    df_sig = pd.DataFrame(records)
    output_path = reports_dir / "baseline_tfidf_signatures_top30.csv"
    df_sig.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

