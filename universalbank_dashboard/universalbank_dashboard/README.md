# UniversalBank — Personal Loan EDA Dashboard (Streamlit)

This repo contains a Streamlit dashboard built from the **UniversalBank** dataset to satisfy the professor's visualization requirements (Income/Age distributions, CCAvg vs Income scatter, ZIP vs Income, correlation heatmap, etc.), with **key insights under every chart**.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

- `data/universalbank_clean.csv` is a cleaned version of the provided CSV (headers fixed + numeric parsing).

## Notes

- Use the **sidebar filters** (Personal Loan, Age, Income, Education, Family) to interactively explore relationships.
- The ZIP visualization shows **top ZIP codes by record count** to reduce clutter (configurable in the sidebar).
