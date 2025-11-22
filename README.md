***Terrorist Group Signature Classification Project***

This project applies supervised machine learning to the Global Terrorism Database (GTD) to classify terrorist organizations based on incident narrative text and structured metadata.

Goals:

- Predict which group committed an attack based solely on the descriptive summary and contextual metadata.
- Extract linguistic and operational signatures that characterize each group’s behavior.

The central question is whether simple text-based models can accurately recover identity and whether those discovered signatures align with real-world geographic, tactical, and ideological patterns. This project demonstrates that terrorist groups exhibit consistent signatures that can be identified through text, enabling accurate classification.

***Groups Included***

This project includes all groups in GTD with more than 100 recorded incidents, excluding:

- "Unknown"
- "Unaffiliated Individual(s)"

This filtering produces a dataset of 70+ organizations with sufficient historical data to learn stable linguistic and operational profiles.

Examples include:

- Taliban
- Islamic State of Iraq and the Levant (ISIL)
- Al-Shabaab
- Communist Party of India – Maoist
- FARC
- Houthi Extremists (Ansar Allah)
- Basque Fatherland and Freedom (ETA)
- New People’s Army (NPA)

***Project Structure***

```
gtd-group-signature/
│
├── data/
│   ├── raw/                     # Original GTD Excel
│   └── processed/               # Preprocessed model ready CSV
│
├── src/
│   ├── preprocess_gtd_group_text.py   # Data cleaning and metadata merging
│   ├── train_baseline_tfidf.py        # TF-IDF + LinearSVC baseline classifier
│   ├── extract_signatures.py          # Top feature extraction per group
│   └── plot_signatures.py             # Signature bar plots and similarity heatmap
│
├── models/                     # Saved baseline model and vectorizer
│
├── reports/
│   ├── baseline_tfidf_val_report.txt
│   ├── baseline_tfidf_test_report.txt
│   ├── baseline_tfidf_signatures_top30.csv
│   └── figures/
│       ├── baseline_tfidf_confusion_matrix_top15.png
│       ├── signature_similarity_top15.png
│       └── signature_top10_<group>.png
│
├── requirements.txt
└── README.md
```

***Installation***

Clone the repository:
```
git clone https://github.com/Jbrog31/gtd-group-signature.git
cd gtd-group-signature
```
Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```
Ensure that the GTD Excel file is placed in:
```
data/raw/GlobalTerrorismDatabase.xlsx
```
***Workflow***

1. Preprocess the GTD data
```
python src/preprocess_gtd_group_text.py
```
This script:

- Removes unusable or short summaries
- Drops "Unknown" and "Unaffiliated" categories
- Keeps groups with >100 incidents
- Combines summary + metadata into a single text field
- Outputs: data/processed/gtd_group_text_subset.csv

2. Train the baseline classifier
```
python src/train_baseline_tfidf.py
```
This step:

- Vectorizes text using TF-IDF (uni/bigrams)
- Trains a LinearSVC classifier
- Generates evaluation outputs
- Saves the model to models/
- Outputs include:
- Validation and test classification reports
- Confusion matrix (top 15 groups)
- Model + vectorizer bundle

3. Extract linguistic and operational signatures
```
python src/extract_signatures.py
```
This script extracts the top 30 most predictive n-grams per group, generating: 
```
reports/baseline_tfidf_signatures_top30.csv
```
This table can be used for signature comparison, audit, or visualization.

4. Visualize signatures and group similarity
```
python src/plot_signatures.py
```
Produces:

- Top-10 signature bar plots for each group
- Signature similarity heatmap for the top 15 groups

These visualizations appear in: 

reports/figures/

***Results Summary***

The classification model achieves extremely strong performance:

- Accuracy: 0.99
- Macro F1-score: 0.98
- Weighted F1-score: 0.99
- Dozens of groups classified with high precision and recall

The model captures consistent organizational signatures reflected in:

- Regional patterns
- Common tactics and targets
- Narrative framing
- Recurrent named locations

Groups such as Taliban, ISIL, Al-Shabaab, CPI-Maoist, and NPA exhibit highly distinct signatures, while smaller or ideologically diffuse groups show slightly more overlap.

Error patterns occur primarily among:

- Groups sharing the same conflict ecosystem
- Islamic State provincial affiliates
- Baloch separatist organizations
- Palestinian militant factions

This aligns closely with real-world relationships.

***Methodology***

This analysis uses the following components:

- Preprocessing of GTD summaries and metadata
- TF-IDF vectorization (unigrams + bigrams)
- LinearSVC classifier with class weighting
- Stratified train/validation/test split
- Signature extraction using model coefficients
- Cosine similarity for inter-group comparison
- Visualizations with matplotlib

***Future Ideas***

Potential enhancements include:

- Fine-tuning a transformer classifier (DistilBERT, MPNet, RoBERTa)
- Building a Streamlit interface for interactive group prediction
- Visualizing embeddings via UMAP or t-SNE
- Incorporating full-text media articles instead of summaries
- Analyzing signature drift over time
- Evaluating model fairness and confusion across geopolitical regions

***Citation and License***

START (National Consortium for the Study of Terrorism and Responses to Terrorism). (2022). Global Terrorism Database, 1970 - 2020 [data file]. https://www.start.umd.edu/data-tools/GTD
Users must comply with GTD’s terms and conditions.
All code in this repository is provided under the MIT License.
