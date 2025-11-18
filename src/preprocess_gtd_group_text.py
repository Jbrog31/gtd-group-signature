from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = raw_dir / "GlobalTerrorismDatabase.xlsx"
    output_path = processed_dir / "gtd_group_text_subset.csv"

    df = pd.read_excel(input_path)

    cols_needed = [
        "eventid",
        "gname",
        "summary",
        "iyear",
        "region_txt",
        "country_txt",
        "attacktype1_txt",
        "targtype1_txt",
    ]
    df = df[cols_needed]

    df = df[
        df["summary"].notna()
        & df["gname"].notna()
        & (df["summary"].str.len() >= 20)
    ]

    df = df[
        ~df["gname"].isin(
            ["Unknown", "Unaffiliated Individual(s)"]
        )
    ]

    group_counts = df["gname"].value_counts()
    groups_to_keep = group_counts[group_counts > 100].index
    df = df[df["gname"].isin(groups_to_keep)]

    df["region_txt"] = df["region_txt"].fillna("UnknownRegion")
    df["country_txt"] = df["country_txt"].fillna("UnknownCountry")
    df["attacktype1_txt"] = df["attacktype1_txt"].fillna("UnknownAttackType")
    df["targtype1_txt"] = df["targtype1_txt"].fillna("UnknownTargetType")
    df["iyear"] = df["iyear"].fillna(-1).astype(int)

    df["text_combined"] = (
        "[REGION: "
        + df["region_txt"].astype(str)
        + "] [COUNTRY: "
        + df["country_txt"].astype(str)
        + "] [ATTACK_TYPE: "
        + df["attacktype1_txt"].astype(str)
        + "] [TARGET: "
        + df["targtype1_txt"].astype(str)
        + "] [YEAR: "
        + df["iyear"].astype(str)
        + "] "
        + df["summary"].astype(str)
    )

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

