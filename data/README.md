# Dataset — CreditWise

## Primary Dataset

**Give Me Some Credit**
Source: [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit/data)

### Download Instructions

1. Log in to Kaggle.
2. Go to: https://www.kaggle.com/c/GiveMeSomeCredit/data
3. Accept the competition rules.
4. Download **`cs-training.csv`** (the labeled training file).
5. Place it in this directory: `data/raw/cs-training.csv`

> **Important**: Do NOT place the unlabeled `cs-test.csv` here.
> Only `cs-training.csv` contains the `SeriousDlqin2yrs` target column
> needed for supervised learning.

### File Details

| File | Rows | Columns | Target |
|---|---|---|---|
| cs-training.csv | ~150,000 | 11 | SeriousDlqin2yrs |

### Target Variable

`SeriousDlqin2yrs`
- 0 = No serious delinquency within 2 years
- 1 = Serious delinquency within 2 years (default)

Expected class distribution: approximately 93% class 0, 7% class 1 (highly imbalanced).

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | float | Credit card balance / credit limits |
| age | int | Borrower's age in years |
| NumberOfTime30-59DaysPastDueNotWorse | int | Count of 30–59 day delinquencies (last 2 years) |
| DebtRatio | float | Monthly debt payments / monthly gross income |
| MonthlyIncome | float | Monthly gross income (USD) — **has missing values** |
| NumberOfOpenCreditLinesAndLoans | int | Total open credit lines + loans |
| NumberOfTimes90DaysLate | int | Times 90+ days past due |
| NumberRealEstateLoansOrLines | int | Real estate loans / lines |
| NumberOfTime60-89DaysPastDueNotWorse | int | Count of 60–89 day delinquencies (last 2 years) |
| NumberOfDependents | int | Number of dependents — **has missing values** |

### Known Data Quality Issues

- `MonthlyIncome`: ~19% missing values → handled with median imputation
- `NumberOfDependents`: ~2.5% missing values → handled with median imputation
- Extreme outliers in `RevolvingUtilizationOfUnsecuredLines` (values >> 1) → log-transform applied
- Extreme values in delinquency columns (96/98 used as missing codes) → investigated during EDA

### Processed Data

After running the training pipeline, processed arrays are saved to `data/processed/`.
These are derived from the raw CSV and should not be committed to version control
(they are excluded by `.gitignore`).

---

## Dataset Licence & Attribution

The Give Me Some Credit dataset is provided by Kaggle for educational and
research purposes. Please review the Kaggle competition terms before use.

CreditWise is an academic prototype. It does NOT use real personal financial
information and does NOT store any applicant data.
