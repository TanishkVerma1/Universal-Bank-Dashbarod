
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="UniversalBank - Personal Loan Dashboard", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "universalbank_clean.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Defensive: normalize expected columns
    df.columns = [c.strip() for c in df.columns]
    return df

def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:.1f}%"

def section_title(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

def key_insights(items):
    st.markdown("**Key insights**")
    for it in items:
        st.markdown(f"- {it}")

df = load_data(DATA_PATH)

# --- column detection (handles minor naming differences) ---
def pick_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

COL_TARGET = pick_col(["Personal_Loan", "Personal Loan", "PersonalLoan"])
COL_INCOME = pick_col(["Income"])
COL_AGE = pick_col(["Age"])
COL_CCAVG = pick_col(["CCAvg", "CC_Avg", "CC_Avg_"])
COL_ZIP = pick_col(["ZIP_Code", "ZIP Code", "Zipcode", "Zip_Code"])
COL_FAMILY = pick_col(["Family"])
COL_MORTGAGE = pick_col(["Mortgage"])
COL_CREDITCARD = pick_col(["CreditCard", "Credit_Card"])
COL_SECURITIES = pick_col(["Securities_Account", "Securities Account"])
COL_CD = pick_col(["CD_Account", "CD Account", "Cash_Deposit", "Cash Deposit"])
COL_EDU = pick_col(["Education"])

missing = [("Personal Loan (target)", COL_TARGET), ("Income", COL_INCOME), ("Age", COL_AGE)]
hard_missing = [name for name, col in missing if col is None]
if hard_missing:
    st.error(f"Dataset is missing required columns: {', '.join(hard_missing)}")
    st.stop()

# --- Sidebar filters ---
st.sidebar.header("Filters")

loan_filter = st.sidebar.selectbox("Personal Loan", ["All", "Yes", "No"], index=0)

age_min, age_max = int(df[COL_AGE].min()), int(df[COL_AGE].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

inc_min, inc_max = float(df[COL_INCOME].min()), float(df[COL_INCOME].max())
income_range = st.sidebar.slider("Income range", float(inc_min), float(inc_max), (float(inc_min), float(inc_max)))

edu_choices = sorted(df[COL_EDU].dropna().unique().tolist()) if COL_EDU else []
if COL_EDU:
    edu_sel = st.sidebar.multiselect("Education", edu_choices, default=edu_choices)
else:
    edu_sel = None

fam_choices = sorted(df[COL_FAMILY].dropna().unique().tolist()) if COL_FAMILY else []
if COL_FAMILY:
    fam_sel = st.sidebar.multiselect("Family size", fam_choices, default=fam_choices)
else:
    fam_sel = None

top_zip_n = st.sidebar.slider("Top ZIP codes (by count) for ZIP plot", 5, 30, 15) if COL_ZIP else 15

# Apply filters
dff = df.copy()

if loan_filter != "All":
    want = 1 if loan_filter == "Yes" else 0
    dff = dff[dff[COL_TARGET] == want]

dff = dff[(dff[COL_AGE] >= age_range[0]) & (dff[COL_AGE] <= age_range[1])]
dff = dff[(dff[COL_INCOME] >= income_range[0]) & (dff[COL_INCOME] <= income_range[1])]

if COL_EDU and edu_sel:
    dff = dff[dff[COL_EDU].isin(edu_sel)]

if COL_FAMILY and fam_sel:
    dff = dff[dff[COL_FAMILY].isin(fam_sel)]

# --- Header KPIs ---
st.title("🏦 UniversalBank — Personal Loan Analytics Dashboard")
st.caption("Exploratory dashboard for understanding which customer attributes are associated with Personal Loan uptake.")

total = len(dff)
loan_rate = (dff[COL_TARGET].mean()) if total else np.nan
avg_inc = dff[COL_INCOME].mean() if total else np.nan
avg_age = dff[COL_AGE].mean() if total else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{total:,}")
c2.metric("Loan Acceptance Rate", fmt_pct(loan_rate))
c3.metric("Avg Income", f"{avg_inc:,.2f}" if total else "—")
c4.metric("Avg Age", f"{avg_age:,.1f}" if total else "—")

st.divider()

# 1) Histograms: Income and Age by Loan
section_title("1) Distribution: Income & Age (Loan vs No Loan)",
             "Overlaid histograms help compare how distributions shift between loan takers and non-takers.")
colA, colB = st.columns(2)

with colA:
    fig = px.histogram(
        dff, x=COL_INCOME, color=COL_TARGET, nbins=30, barmode="overlay",
        labels={COL_TARGET: "Personal Loan", COL_INCOME: "Income"},
        category_orders={COL_TARGET: [0, 1]}
    )
    fig.update_layout(legend_title_text="Personal Loan", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    if total:
        inc_yes = dff.loc[dff[COL_TARGET] == 1, COL_INCOME].mean()
        inc_no = dff.loc[dff[COL_TARGET] == 0, COL_INCOME].mean()
        key_insights([
            f"Average income is **{inc_yes:,.1f}** for loan takers vs **{inc_no:,.1f}** for non-takers." if not (np.isnan(inc_yes) or np.isnan(inc_no)) else "Not enough class data to compare averages.",
            "Look for a right-shift (higher income) in the loan-taker distribution to indicate stronger association."
        ])

with colB:
    fig = px.histogram(
        dff, x=COL_AGE, color=COL_TARGET, nbins=25, barmode="overlay",
        labels={COL_TARGET: "Personal Loan", COL_AGE: "Age"},
        category_orders={COL_TARGET: [0, 1]}
    )
    fig.update_layout(legend_title_text="Personal Loan", height=420)
    st.plotly_chart(fig, use_container_width=True)

    if total:
        age_yes = dff.loc[dff[COL_TARGET] == 1, COL_AGE].mean()
        age_no = dff.loc[dff[COL_TARGET] == 0, COL_AGE].mean()
        key_insights([
            f"Average age is **{age_yes:,.1f}** for loan takers vs **{age_no:,.1f}** for non-takers." if not (np.isnan(age_yes) or np.isnan(age_no)) else "Not enough class data to compare averages.",
            "If both curves overlap heavily, age alone may be a weak discriminator."
        ])

st.divider()

# 2) Scatter: CCAvg vs Income (wrt Loan)
section_title("2) Spending vs Income (CCAvg vs Income) by Personal Loan",
             "Scatter plot shows how credit card spending (CCAvg) relates to income, split by loan acceptance.")
if COL_CCAVG:
    fig = px.scatter(
        dff, x=COL_INCOME, y=COL_CCAVG, color=COL_TARGET,
        hover_data=[COL_AGE] + ([COL_EDU] if COL_EDU else []),
        labels={COL_TARGET: "Personal Loan", COL_INCOME: "Income", COL_CCAVG: "CCAvg"},
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    if total:
        corr = dff[[COL_INCOME, COL_CCAVG]].corr().iloc[0,1]
        cc_yes = dff.loc[dff[COL_TARGET]==1, COL_CCAVG].median()
        cc_no = dff.loc[dff[COL_TARGET]==0, COL_CCAVG].median()
        key_insights([
            f"Income–CCAvg correlation in filtered data: **{corr:.2f}** (closer to 1 means stronger positive relationship).",
            f"Median CCAvg is **{cc_yes:,.2f}** for loan takers vs **{cc_no:,.2f}** for non-takers." if not (np.isnan(cc_yes) or np.isnan(cc_no)) else "Not enough class data to compare medians.",
            "Clusters with **higher CCAvg at similar incomes** can indicate spending behavior associated with loan acceptance."
        ])
else:
    st.warning("CCAvg column not found in the dataset, so this plot is skipped.")

st.divider()

# 3) ZIP Code vs Income vs Personal Loan
section_title("3) ZIP Code vs Income (colored by Loan)",
             "ZIP codes are many; we show the top ZIPs by count (filter on sidebar).")
if COL_ZIP:
    top_zips = dff[COL_ZIP].value_counts().head(top_zip_n).index.tolist()
    dzip = dff[dff[COL_ZIP].isin(top_zips)].copy()

    fig = px.strip(
        dzip, x=COL_ZIP, y=COL_INCOME, color=COL_TARGET,
        labels={COL_TARGET: "Personal Loan", COL_ZIP: "ZIP Code", COL_INCOME: "Income"},
        stripmode="overlay"
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    if len(dzip):
        zip_rates = dzip.groupby(COL_ZIP)[COL_TARGET].mean().sort_values(ascending=False)
        best_zip = zip_rates.index[0]
        worst_zip = zip_rates.index[-1]
        key_insights([
            f"Among top ZIPs, highest loan acceptance is **{best_zip}** at **{zip_rates.iloc[0]*100:.1f}%**.",
            f"Lowest acceptance is **{worst_zip}** at **{zip_rates.iloc[-1]*100:.1f}%**.",
            "ZIP can proxy location/affluence—use it carefully (risk of spurious correlation)."
        ])
else:
    st.warning("ZIP Code column not found in the dataset, so this plot is skipped.")

st.divider()

# 4) Correlation Heatmap (all numeric columns)
section_title("4) Correlation Heatmap (All Numeric Columns)",
             "Helps identify variables that move together; strong correlations may indicate redundancy or strong predictors.")
num = dff.select_dtypes(include=[np.number]).copy()
if len(num.columns) >= 2:
    corr = num.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        labels=dict(color="Correlation")
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    # Insights: top correlations with target (exclude self)
    if COL_TARGET in corr.columns:
        targ = corr[COL_TARGET].drop(COL_TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
        top = targ.head(5)
        items = [f"Top correlation with **Personal Loan**: **{top.index[0]}** (r = {top.iloc[0]:.2f})."] if len(top) else ["No correlations available."]
        items += [f"Next: **{idx}** (r = {val:.2f})" for idx, val in top.iloc[1:].items()]
        items.append("Correlation ≠ causation; use this to guide feature exploration and modeling.")
        key_insights(items)
else:
    st.warning("Not enough numeric columns to compute correlations.")

st.divider()

# 5) Family size vs Income vs Mortgage/CCAvg vs Personal Loan
section_title("5) Family, Income, Mortgage & CCAvg vs Personal Loan",
             "A multi-dimensional bubble plot: Income vs Mortgage, bubble size=CCAvg, colored by Personal Loan.")
if COL_FAMILY and COL_MORTGAGE and COL_CCAVG:
    fam_focus = st.selectbox("Focus Family size (optional)", ["All"] + fam_choices, index=0)
    df5 = dff if fam_focus == "All" else dff[dff[COL_FAMILY] == fam_focus]
    fig = px.scatter(
        df5, x=COL_INCOME, y=COL_MORTGAGE, size=COL_CCAVG, color=COL_TARGET,
        hover_data=[COL_FAMILY] + ([COL_EDU] if COL_EDU else []),
        labels={COL_TARGET: "Personal Loan", COL_INCOME: "Income", COL_MORTGAGE: "Mortgage", COL_CCAVG: "CCAvg", COL_FAMILY: "Family"}
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

    if len(df5):
        # Loan rate by family
        fam_rate = df5.groupby(COL_FAMILY)[COL_TARGET].mean().sort_values(ascending=False)
        best_f = fam_rate.index[0]
        key_insights([
            f"Highest loan acceptance family size (in current filter): **{best_f}** at **{fam_rate.iloc[0]*100:.1f}%**.",
            "Higher **Income + higher CCAvg** bubbles often align with loan takers in many retail-banking datasets.",
            "If mortgage is high but loan acceptance is low, it may indicate already-leveraged customers."
        ])
else:
    st.warning("One or more required columns (Family, Mortgage, CCAvg) missing; this visualization is skipped.")

st.divider()

# 6) Securities vs CD/Cash Deposit vs CreditCard vs Personal Loan
section_title("6) Product Holding vs Personal Loan",
             "Compares how ownership of products relates to Personal Loan acceptance.")
cols_needed = [c for c in [COL_SECURITIES, COL_CD, COL_CREDITCARD] if c is not None]
if cols_needed:
    # Build a tidy table: feature, value, loan_rate, count
    tidy = []
    for feat in cols_needed:
        tmp = dff.groupby(feat)[COL_TARGET].agg(["mean", "count"]).reset_index()
        tmp["feature"] = feat
        tmp.rename(columns={feat: "has_product", "mean": "loan_rate", "count": "n"}, inplace=True)
        tidy.append(tmp)
    tidy = pd.concat(tidy, ignore_index=True)

    fig = px.bar(
        tidy, x="feature", y="loan_rate", color="has_product",
        barmode="group", text=tidy["loan_rate"].map(lambda x: f"{x*100:.1f}%"),
        labels={"feature": "Product", "loan_rate": "Loan acceptance rate", "has_product": "Has product"}
    )
    fig.update_layout(height=520, yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    items = []
    for feat in cols_needed:
        tmp = dff.groupby(feat)[COL_TARGET].mean()
        if 1 in tmp.index and 0 in tmp.index:
            items.append(f"**{feat}**: acceptance is **{tmp.loc[1]*100:.1f}%** (has=1) vs **{tmp.loc[0]*100:.1f}%** (has=0).")
    items.append("If product-holders have higher acceptance, they may be more engaged/affluent—use for targeting.")
    key_insights(items)
else:
    st.warning("Required product columns not found (Securities/CD/CreditCard).")

st.divider()

# 7) Box & Whisker: CreditCard, CCAvg and Income
section_title("7) Box Plots: Income & CCAvg by Credit Card Ownership",
             "Box plots highlight median, spread, and outliers for each group (CreditCard=0 vs 1).")
if COL_CREDITCARD:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            dff, x=COL_CREDITCARD, y=COL_INCOME, points="outliers",
            labels={COL_CREDITCARD: "Has Credit Card", COL_INCOME: "Income"}
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        if total:
            med0 = dff.loc[dff[COL_CREDITCARD]==0, COL_INCOME].median()
            med1 = dff.loc[dff[COL_CREDITCARD]==1, COL_INCOME].median()
            key_insights([f"Median income: **{med1:,.1f}** (CreditCard=1) vs **{med0:,.1f}** (CreditCard=0)."])
    with col2:
        if COL_CCAVG:
            fig = px.box(
                dff, x=COL_CREDITCARD, y=COL_CCAVG, points="outliers",
                labels={COL_CREDITCARD: "Has Credit Card", COL_CCAVG: "CCAvg"}
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
            if total:
                med0 = dff.loc[dff[COL_CREDITCARD]==0, COL_CCAVG].median()
                med1 = dff.loc[dff[COL_CREDITCARD]==1, COL_CCAVG].median()
                key_insights([f"Median CCAvg: **{med1:,.2f}** (CreditCard=1) vs **{med0:,.2f}** (CreditCard=0)."])
        else:
            st.warning("CCAvg not available for the second box plot.")
else:
    st.warning("CreditCard column not found; this section is skipped.")

st.divider()

# 8) Education vs Income vs Personal Loan
section_title("8) Education vs Income vs Personal Loan",
             "Compare income distribution and loan acceptance across education levels.")
if COL_EDU:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig = px.box(
            dff, x=COL_EDU, y=COL_INCOME, color=COL_TARGET,
            labels={COL_EDU: "Education", COL_INCOME: "Income", COL_TARGET: "Personal Loan"}
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        edu_rate = dff.groupby(COL_EDU)[COL_TARGET].mean().reset_index()
        fig = px.bar(
            edu_rate, x=COL_EDU, y=COL_TARGET, text=edu_rate[COL_TARGET].map(lambda x: f"{x*100:.1f}%"),
            labels={COL_EDU: "Education", COL_TARGET: "Loan acceptance rate"}
        )
        fig.update_layout(height=520, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    if len(dff):
        best = edu_rate.sort_values(COL_TARGET, ascending=False).iloc[0]
        key_insights([
            f"Education level **{best[COL_EDU]}** has the highest acceptance rate at **{best[COL_TARGET]*100:.1f}%**.",
            "If education segments have different income ranges, interpret acceptance differences together with income."
        ])
else:
    st.warning("Education column not found; this section is skipped.")

st.divider()

# 9) Mortgage vs Income vs Family size vs Personal Loan
section_title("9) Mortgage vs Income vs Family vs Personal Loan",
             "3D scatter helps see joint effects; use filters to reduce clutter.")
if COL_MORTGAGE and COL_FAMILY:
    sample_n = st.slider("Sample size for 3D plot (performance)", 200, min(2000, len(dff)), min(1200, len(dff)))
    d3 = dff.sample(sample_n, random_state=42) if len(dff) > sample_n else dff

    fig = px.scatter_3d(
        d3, x=COL_INCOME, y=COL_MORTGAGE, z=COL_FAMILY, color=COL_TARGET,
        labels={COL_INCOME: "Income", COL_MORTGAGE: "Mortgage", COL_FAMILY: "Family", COL_TARGET: "Personal Loan"}
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    if len(dff):
        # simple insight: loan rate by mortgage bucket
        bins = pd.qcut(dff[COL_MORTGAGE].rank(method="first"), 4, labels=["Low", "Mid-Low", "Mid-High", "High"])
        tmp = dff.groupby(bins)[COL_TARGET].mean()
        key_insights([
            "Loan acceptance by mortgage quartile: " + ", ".join([f"**{k}** {v*100:.1f}%" for k, v in tmp.items()]),
            "If high-mortgage customers have lower acceptance, they may already be credit-constrained."
        ])
else:
    st.warning("Mortgage and/or Family columns not found; this section is skipped.")

st.divider()

st.markdown("### Notes")
st.write(
    "This dashboard is designed for **EDA (exploratory data analysis)**. "
    "If you later need a predictive model, these insights can guide feature selection and threshold decisions."
)
