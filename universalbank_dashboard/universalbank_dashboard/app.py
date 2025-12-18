import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)


st.set_page_config(page_title="UniversalBank - Personal Loan Dashboard", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "universalbank_clean.csv"

# -----------------------------
# Global styling (better contrast)
# -----------------------------
px.defaults.template = "plotly_dark"

# High-contrast, colorblind-friendly 2-class palette
LOAN_COLOR_MAP = {
    "No": "#1f77b4",   # blue
    "Yes": "#ff7f0e",  # orange
}

# For binary product ownership bars (0/1)
BINARY_COLOR_MAP = {
    0: "#7f7f7f",      # gray
    1: "#2ca02c",      # green
}

# Heatmap diverging palette (clear negative vs positive)
CORR_SCALE = "RdBu"  # diverging

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
income_range = st.sidebar.slider(
    "Income range",
    float(inc_min),
    float(inc_max),
    (float(inc_min), float(inc_max)),
)

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

# Create readable labels for color (this improves legend + consistency)
LOAN_LABEL_COL = "_loan_label"
if COL_TARGET in dff.columns:
    dff[LOAN_LABEL_COL] = dff[COL_TARGET].map({0: "No", 1: "Yes"}).astype("category")

st.sidebar.markdown("---")
page = st.sidebar.radio("Pages", ["📊 EDA Dashboard", "🧠 Segmentation (K-Means)", "🤖 Classification Models"], index=0)

if page == "📊 EDA Dashboard":

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
    section_title(
        "1) Distribution: Income & Age (Loan vs No Loan)",
        "Overlaid histograms help compare how distributions shift between loan takers and non-takers.",
    )
    colA, colB = st.columns(2)
    
    with colA:
        fig = px.histogram(
            dff,
            x=COL_INCOME,
            color=LOAN_LABEL_COL,
            nbins=30,
            barmode="overlay",
            labels={LOAN_LABEL_COL: "Personal Loan", COL_INCOME: "Income"},
            category_orders={LOAN_LABEL_COL: ["No", "Yes"]},
            color_discrete_map=LOAN_COLOR_MAP,
            opacity=0.55,  # helps overlaps stay distinguishable
        )
        fig.update_layout(legend_title_text="Personal Loan", height=420)
        st.plotly_chart(fig, use_container_width=True)
    
        if total:
            inc_yes = dff.loc[dff[COL_TARGET] == 1, COL_INCOME].mean()
            inc_no = dff.loc[dff[COL_TARGET] == 0, COL_INCOME].mean()
            key_insights([
                f"Average income is **{inc_yes:,.1f}** for loan takers vs **{inc_no:,.1f}** for non-takers."
                if not (np.isnan(inc_yes) or np.isnan(inc_no))
                else "Not enough class data to compare averages.",
                "A clear right-shift (higher income) for **Yes** suggests income is strongly associated with acceptance.",
            ])
    
    with colB:
        fig = px.histogram(
            dff,
            x=COL_AGE,
            color=LOAN_LABEL_COL,
            nbins=25,
            barmode="overlay",
            labels={LOAN_LABEL_COL: "Personal Loan", COL_AGE: "Age"},
            category_orders={LOAN_LABEL_COL: ["No", "Yes"]},
            color_discrete_map=LOAN_COLOR_MAP,
            opacity=0.55,
        )
        fig.update_layout(legend_title_text="Personal Loan", height=420)
        st.plotly_chart(fig, use_container_width=True)
    
        if total:
            age_yes = dff.loc[dff[COL_TARGET] == 1, COL_AGE].mean()
            age_no = dff.loc[dff[COL_TARGET] == 0, COL_AGE].mean()
            key_insights([
                f"Average age is **{age_yes:,.1f}** for loan takers vs **{age_no:,.1f}** for non-takers."
                if not (np.isnan(age_yes) or np.isnan(age_no))
                else "Not enough class data to compare averages.",
                "If distributions overlap heavily, age may be a weaker discriminator than income/spend.",
            ])
    
    st.divider()
    
    # 2) Scatter: CCAvg vs Income (wrt Loan)
    section_title(
        "2) Spending vs Income (CCAvg vs Income) by Personal Loan",
        "Scatter plot shows how credit card spending (CCAvg) relates to income, split by loan acceptance.",
    )
    if COL_CCAVG:
        fig = px.scatter(
            dff,
            x=COL_INCOME,
            y=COL_CCAVG,
            color=LOAN_LABEL_COL,
            symbol=LOAN_LABEL_COL,  # extra channel helps distinguish points
            hover_data=[COL_AGE] + ([COL_EDU] if COL_EDU else []),
            labels={LOAN_LABEL_COL: "Personal Loan", COL_INCOME: "Income", COL_CCAVG: "CCAvg"},
            color_discrete_map=LOAN_COLOR_MAP,
            opacity=0.75,
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="rgba(255,255,255,0.35)")))
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    
        if total:
            corr = dff[[COL_INCOME, COL_CCAVG]].corr().iloc[0, 1]
            cc_yes = dff.loc[dff[COL_TARGET] == 1, COL_CCAVG].median()
            cc_no = dff.loc[dff[COL_TARGET] == 0, COL_CCAVG].median()
            key_insights([
                f"Income–CCAvg correlation in filtered data: **{corr:.2f}**.",
                f"Median CCAvg is **{cc_yes:,.2f}** for loan takers vs **{cc_no:,.2f}** for non-takers."
                if not (np.isnan(cc_yes) or np.isnan(cc_no))
                else "Not enough class data to compare medians.",
                "If **Yes** points concentrate at higher CCAvg for similar income, spending behavior may be a strong signal.",
            ])
    else:
        st.warning("CCAvg column not found in the dataset, so this plot is skipped.")
    
    st.divider()
    
    # 3) ZIP Code vs Income vs Personal Loan
    section_title(
        "3) ZIP Code vs Income (colored by Loan)",
        "ZIP codes are many; we show the top ZIPs by count (filter on sidebar).",
    )
    if COL_ZIP:
        top_zips = dff[COL_ZIP].value_counts().head(top_zip_n).index.tolist()
        dzip = dff[dff[COL_ZIP].isin(top_zips)].copy()
    
        fig = px.strip(
            dzip,
            x=COL_ZIP,
            y=COL_INCOME,
            color=LOAN_LABEL_COL,
            labels={LOAN_LABEL_COL: "Personal Loan", COL_ZIP: "ZIP Code", COL_INCOME: "Income"},
            stripmode="overlay",
            color_discrete_map=LOAN_COLOR_MAP,
        )
        fig.update_traces(jitter=0.35, marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color="rgba(255,255,255,0.35)")))
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    
        if len(dzip):
            zip_rates = dzip.groupby(COL_ZIP)[COL_TARGET].mean().sort_values(ascending=False)
            best_zip = zip_rates.index[0]
            worst_zip = zip_rates.index[-1]
            key_insights([
                f"Among top ZIPs, highest loan acceptance is **{best_zip}** at **{zip_rates.iloc[0]*100:.1f}%**.",
                f"Lowest acceptance is **{worst_zip}** at **{zip_rates.iloc[-1]*100:.1f}%**.",
                "ZIP can proxy location/affluence—interpret carefully and avoid over-claiming causality.",
            ])
    else:
        st.warning("ZIP Code column not found in the dataset, so this plot is skipped.")
    
    st.divider()
    
    # 4) Correlation Heatmap (all numeric columns)
    section_title(
        "4) Correlation Heatmap (All Numeric Columns)",
        "Helps identify variables that move together; strong correlations may indicate redundancy or strong predictors.",
    )
    num = dff.select_dtypes(include=[np.number]).copy()
    if len(num.columns) >= 2:
        corr = num.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=CORR_SCALE,  # diverging -> clearer
            zmin=-1,
            zmax=1,
            labels=dict(color="Correlation"),
        )
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
    
        if COL_TARGET in corr.columns:
            targ = corr[COL_TARGET].drop(COL_TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
            top = targ.head(5)
            items = [f"Top correlation with **Personal Loan**: **{top.index[0]}** (r = {top.iloc[0]:.2f})."] if len(top) else ["No correlations available."]
            items += [f"Next: **{idx}** (r = {val:.2f})" for idx, val in top.iloc[1:].items()]
            items.append("Correlation ≠ causation; use this mainly to guide feature exploration.")
            key_insights(items)
    else:
        st.warning("Not enough numeric columns to compute correlations.")
    
    st.divider()
    
    # 5) Family size vs Income vs Mortgage/CCAvg vs Personal Loan
    section_title(
        "5) Family, Income, Mortgage & CCAvg vs Personal Loan",
        "A multi-dimensional bubble plot: Income vs Mortgage, bubble size=CCAvg, colored by Personal Loan.",
    )
    if COL_FAMILY and COL_MORTGAGE and COL_CCAVG:
        fam_focus = st.selectbox("Focus Family size (optional)", ["All"] + fam_choices, index=0)
        df5 = dff if fam_focus == "All" else dff[dff[COL_FAMILY] == fam_focus]
    
        fig = px.scatter(
            df5,
            x=COL_INCOME,
            y=COL_MORTGAGE,
            size=COL_CCAVG,
            color=LOAN_LABEL_COL,
            symbol=LOAN_LABEL_COL,
            hover_data=[COL_FAMILY] + ([COL_EDU] if COL_EDU else []),
            labels={
                LOAN_LABEL_COL: "Personal Loan",
                COL_INCOME: "Income",
                COL_MORTGAGE: "Mortgage",
                COL_CCAVG: "CCAvg",
                COL_FAMILY: "Family",
            },
            color_discrete_map=LOAN_COLOR_MAP,
            opacity=0.75,
            size_max=28,
        )
        fig.update_traces(marker=dict(line=dict(width=0.6, color="rgba(255,255,255,0.35)")))
        fig.update_layout(height=560)
        st.plotly_chart(fig, use_container_width=True)
    
        if len(df5):
            fam_rate = df5.groupby(COL_FAMILY)[COL_TARGET].mean().sort_values(ascending=False)
            best_f = fam_rate.index[0]
            key_insights([
                f"Highest loan acceptance family size (in current filter): **{best_f}** at **{fam_rate.iloc[0]*100:.1f}%**.",
                "Higher **Income + higher CCAvg** bubbles often align with loan takers in retail-banking datasets.",
                "If mortgage is high but loan acceptance is low, it may indicate already-leveraged customers.",
            ])
    else:
        st.warning("One or more required columns (Family, Mortgage, CCAvg) missing; this visualization is skipped.")
    
    st.divider()
    
    # 6) Securities vs CD/Cash Deposit vs CreditCard vs Personal Loan
    section_title(
        "6) Product Holding vs Personal Loan",
        "Compares how ownership of products relates to Personal Loan acceptance.",
    )
    cols_needed = [c for c in [COL_SECURITIES, COL_CD, COL_CREDITCARD] if c is not None]
    if cols_needed:
        tidy = []
        for feat in cols_needed:
            tmp = dff.groupby(feat)[COL_TARGET].agg(["mean", "count"]).reset_index()
            tmp["feature"] = feat
            tmp.rename(columns={feat: "has_product", "mean": "loan_rate", "count": "n"}, inplace=True)
            tidy.append(tmp)
        tidy = pd.concat(tidy, ignore_index=True)
    
        fig = px.bar(
            tidy,
            x="feature",
            y="loan_rate",
            color="has_product",
            barmode="group",
            text=tidy["loan_rate"].map(lambda x: f"{x*100:.1f}%"),
            labels={"feature": "Product", "loan_rate": "Loan acceptance rate", "has_product": "Has product"},
            color_discrete_map=BINARY_COLOR_MAP,  # clearer 0 vs 1
        )
        fig.update_layout(height=520, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    
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
    section_title(
        "7) Box Plots: Income & CCAvg by Credit Card Ownership",
        "Box plots highlight median, spread, and outliers for each group (CreditCard=0 vs 1).",
    )
    if COL_CREDITCARD:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(
                dff,
                x=COL_CREDITCARD,
                y=COL_INCOME,
                points="outliers",
                labels={COL_CREDITCARD: "Has Credit Card", COL_INCOME: "Income"},
                color=COL_CREDITCARD,
                color_discrete_map=BINARY_COLOR_MAP,
            )
            fig.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            if total:
                med0 = dff.loc[dff[COL_CREDITCARD] == 0, COL_INCOME].median()
                med1 = dff.loc[dff[COL_CREDITCARD] == 1, COL_INCOME].median()
                key_insights([f"Median income: **{med1:,.1f}** (CreditCard=1) vs **{med0:,.1f}** (CreditCard=0)."])
    
        with col2:
            if COL_CCAVG:
                fig = px.box(
                    dff,
                    x=COL_CREDITCARD,
                    y=COL_CCAVG,
                    points="outliers",
                    labels={COL_CREDITCARD: "Has Credit Card", COL_CCAVG: "CCAvg"},
                    color=COL_CREDITCARD,
                    color_discrete_map=BINARY_COLOR_MAP,
                )
                fig.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                if total:
                    med0 = dff.loc[dff[COL_CREDITCARD] == 0, COL_CCAVG].median()
                    med1 = dff.loc[dff[COL_CREDITCARD] == 1, COL_CCAVG].median()
                    key_insights([f"Median CCAvg: **{med1:,.2f}** (CreditCard=1) vs **{med0:,.2f}** (CreditCard=0)."])
            else:
                st.warning("CCAvg not available for the second box plot.")
    else:
        st.warning("CreditCard column not found; this section is skipped.")
    
    st.divider()
    
    # 8) Education vs Income vs Personal Loan
    section_title(
        "8) Education vs Income vs Personal Loan",
        "Compare income distribution and loan acceptance across education levels.",
    )
    if COL_EDU:
        col1, col2 = st.columns([1.2, 1])
        with col1:
            fig = px.box(
                dff,
                x=COL_EDU,
                y=COL_INCOME,
                color=LOAN_LABEL_COL,
                labels={COL_EDU: "Education", COL_INCOME: "Income", LOAN_LABEL_COL: "Personal Loan"},
                color_discrete_map=LOAN_COLOR_MAP,
            )
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
    
        with col2:
            edu_rate = dff.groupby(COL_EDU)[COL_TARGET].mean().reset_index()
            fig = px.bar(
                edu_rate,
                x=COL_EDU,
                y=COL_TARGET,
                text=edu_rate[COL_TARGET].map(lambda x: f"{x*100:.1f}%"),
                labels={COL_EDU: "Education", COL_TARGET: "Loan acceptance rate"},
            )
            fig.update_layout(height=520, yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
    
        if len(dff):
            best = edu_rate.sort_values(COL_TARGET, ascending=False).iloc[0]
            key_insights([
                f"Education level **{best[COL_EDU]}** has the highest acceptance rate at **{best[COL_TARGET]*100:.1f}%**.",
                "Interpret education differences alongside income/spend (education may correlate with both).",
            ])
    else:
        st.warning("Education column not found; this section is skipped.")
    
    st.divider()
    
    # 9) Mortgage vs Income vs Family size vs Personal Loan
    section_title(
        "9) Mortgage vs Income vs Family vs Personal Loan",
        "3D scatter helps see joint effects; use filters to reduce clutter.",
    )
    if COL_MORTGAGE and COL_FAMILY:
        sample_n = st.slider("Sample size for 3D plot (performance)", 200, min(2000, len(dff)), min(1200, len(dff)))
        d3 = dff.sample(sample_n, random_state=42) if len(dff) > sample_n else dff
    
        fig = px.scatter_3d(
            d3,
            x=COL_INCOME,
            y=COL_MORTGAGE,
            z=COL_FAMILY,
            color=LOAN_LABEL_COL,
            labels={COL_INCOME: "Income", COL_MORTGAGE: "Mortgage", COL_FAMILY: "Family", LOAN_LABEL_COL: "Personal Loan"},
            color_discrete_map=LOAN_COLOR_MAP,
            opacity=0.75,
        )
        fig.update_traces(marker=dict(size=4, line=dict(width=0.4, color="rgba(255,255,255,0.35)")))
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
    
        if len(dff):
            bins = pd.qcut(dff[COL_MORTGAGE].rank(method="first"), 4, labels=["Low", "Mid-Low", "Mid-High", "High"])
            tmp = dff.groupby(bins)[COL_TARGET].mean()
            key_insights([
                "Loan acceptance by mortgage quartile: " + ", ".join([f"**{k}** {v*100:.1f}%" for k, v in tmp.items()]),
                "If high-mortgage customers have lower acceptance, they may already be credit-constrained.",
            ])
    else:
        st.warning("Mortgage and/or Family columns not found; this section is skipped.")
    
    st.divider()
    
    st.markdown("### Notes")
    st.write(
        "This dashboard is designed for **EDA (exploratory data analysis)**. "
        "If you later need a predictive model, these insights can guide feature selection and threshold decisions."
    )
# =========================================================
# NEW PAGE 1: K-MEANS CLUSTERING (CUSTOMER SEGMENTATION)
# =========================================================
if page == "🧠 Segmentation (K-Means)":

    st.title("🧠 Customer Segmentation using K-Means")
    st.caption("Elbow method + Interactive 3D clustering plot (k=3) with centroids and boundary ellipsoids.")

    # --- Feature set for clustering (numeric only, scaled) ---
    cluster_features = [COL_AGE, COL_INCOME]
    if COL_CCAVG: cluster_features.append(COL_CCAVG)
    if COL_MORTGAGE: cluster_features.append(COL_MORTGAGE)
    if COL_FAMILY: cluster_features.append(COL_FAMILY)

    # Remove None
    cluster_features = [c for c in cluster_features if c is not None]

    if len(cluster_features) < 3:
        st.error("Not enough numeric features found for clustering (need at least 3).")
        st.stop()

    Xc = dff[cluster_features].copy()
    Xc = Xc.dropna()

    scaler = StandardScaler()
    Xc_scaled = scaler.fit_transform(Xc)

    st.subheader("1) Elbow Method (Inertia vs K)")
    k_max = st.slider("Max K to test (Elbow)", 5, 15, 10)
    inertias = []
    ks = list(range(1, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xc_scaled)
        inertias.append(km.inertia_)

    fig_elbow = px.line(
        x=ks, y=inertias, markers=True,
        labels={"x": "Number of clusters (K)", "y": "Inertia (Within-cluster SSE)"},
        title="Elbow Method"
    )
    fig_elbow.update_layout(height=420)
    st.plotly_chart(fig_elbow, use_container_width=True)

    st.markdown("**Key insights**")
    st.markdown("- The elbow point (where inertia reduction starts slowing) is a good choice for K.")
    st.markdown("- Choose a K that balances model simplicity and cluster separation.")

    st.divider()
    st.subheader("2) Interactive 3D Clustering (k=3) with Centroids + Boundaries")

    k_choice = st.selectbox("Select K for clustering view", [3, 4, 5], index=0)

    km = KMeans(n_clusters=k_choice, random_state=42, n_init=10)
    labels_k = km.fit_predict(Xc_scaled)

    # Create plotting DF (use first 3 features for 3D)
    f1, f2, f3 = cluster_features[:3]
    X_plot = Xc[[f1, f2, f3]].copy()
    X_plot["cluster"] = labels_k

    # Centroids in original scale (for all features), then slice to 3D
    centers_scaled = km.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)
    cent_df = pd.DataFrame(centers_orig, columns=cluster_features)
    cent_df = cent_df[[f1, f2, f3]]
    cent_df["cluster"] = list(range(k_choice))

    # Main 3D scatter
    fig3 = px.scatter_3d(
        X_plot, x=f1, y=f2, z=f3,
        color="cluster",
        opacity=0.75,
        title=f"K-Means Clusters (K={k_choice}) — Features: {f1}, {f2}, {f3}",
        hover_data=[COL_TARGET] if COL_TARGET in Xc.columns else None
    )

    # Add centroids
    fig3.add_trace(go.Scatter3d(
        x=cent_df[f1], y=cent_df[f2], z=cent_df[f3],
        mode="markers+text",
        marker=dict(size=10, symbol="diamond", color="white"),
        text=[f"C{i}" for i in cent_df["cluster"]],
        textposition="top center",
        name="Centroids"
    ))

    # --- “Boundary” ellipsoids (approx cluster boundary using covariance in original space for 3 features) ---
    def ellipsoid_surface(center, cov, n=25, scale=1.8):
        # create ellipsoid surface using eigen decomposition
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        radii = scale * np.sqrt(vals)

        u = np.linspace(0, 2*np.pi, n)
        v = np.linspace(0, np.pi, n)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # scale
        pts = np.stack([x, y, z], axis=-1)
        pts = pts * radii

        # rotate
        pts = pts @ vecs.T

        Xs = pts[..., 0] + center[0]
        Ys = pts[..., 1] + center[1]
        Zs = pts[..., 2] + center[2]
        return Xs, Ys, Zs

    if k_choice == 3:
        # compute cov per cluster on the 3 plotted features
        for cl in range(3):
            pts = X_plot[X_plot["cluster"] == cl][[f1, f2, f3]].values
            if len(pts) < 5:
                continue
            cov = np.cov(pts.T)
            center = cent_df[cent_df["cluster"] == cl][[f1, f2, f3]].values.flatten()

            Xs, Ys, Zs = ellipsoid_surface(center, cov, n=26, scale=1.8)
            fig3.add_trace(go.Surface(
                x=Xs, y=Ys, z=Zs,
                opacity=0.18,
                showscale=False,
                name=f"Boundary C{cl}"
            ))

    fig3.update_layout(height=700)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Key insights**")
    st.markdown("- Each cluster represents a customer segment with similar financial/behavior patterns.")
    st.markdown("- Centroids show the “typical” customer of each segment.")
    st.markdown("- Boundary ellipsoids give an approximate visual separation of clusters in 3D feature space.")

    st.divider()

    # Extra requested check: CCAvg > 0 but CreditCard = 0
    st.subheader("📌 Data Quality / Behavior Check")
    if COL_CCAVG and COL_CREDITCARD:
        odd = dff[(dff[COL_CCAVG] > 0) & (dff[COL_CREDITCARD] == 0)].copy()
        st.write(f"Customers with **CCAvg > 0** but **CreditCard = 0**: **{len(odd):,}**")

        if len(odd) > 0:
            has_mort = (odd[COL_MORTGAGE] > 0).mean() if COL_MORTGAGE else np.nan
            has_loan = odd[COL_TARGET].mean()
            key_insights([
                f"Among these customers, **{has_loan*100:.1f}%** accepted personal loan (Yes).",
                f"Among these customers, **{has_mort*100:.1f}%** have mortgage > 0." if COL_MORTGAGE else "Mortgage column not available.",
                "This pattern may indicate data entry issues or that CCAvg represents spending estimate even without explicit credit-card ownership."
            ])

            st.dataframe(odd.head(20), use_container_width=True)
    else:
        st.info("CCAvg or CreditCard column missing — cannot run this check.")


# =========================================================
# NEW PAGE 2: CLASSIFICATION MODELS (COMPARE 4 ALGORITHMS)
# =========================================================
if page == "🤖 Classification Models":

    st.title("🤖 Classification Models — Predict Personal Loan Acceptance")
    st.caption("Models: KNN, Decision Tree, Random Forest, Gradient Boosted Tree. Positive class = Yes (1).")

    # ---------- Build modeling dataset ----------
    # Use numeric + Education (treated as numeric levels) and product indicators
    model_cols = []
    for c in [COL_AGE, COL_INCOME, COL_CCAVG, COL_MORTGAGE, COL_FAMILY, COL_EDU, COL_CREDITCARD, COL_SECURITIES, COL_CD]:
        if c is not None and c in dff.columns:
            model_cols.append(c)

    # Remove ZIP because high-cardinality noise; also remove target itself
    X = dff[model_cols].copy()
    y = dff[COL_TARGET].astype(int).copy()  # 1 = Yes, 0 = No

    # drop missing rows
    m = pd.concat([X, y], axis=1).dropna()
    X = m[model_cols]
    y = m[COL_TARGET]

    st.subheader("Dataset Used For Modeling")
    st.write(f"Rows after filtering + dropping missing: **{len(X):,}** | Features used: **{len(model_cols)}**")
    st.write("Positive class: **Personal Loan = Yes (1)**")

    test_size = st.slider("Test size", 0.2, 0.4, 0.3, 0.05)
    random_state = st.number_input("Random seed", 0, 9999, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # ---------- Models ----------
    models = {
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state),
        "Gradient Boosted Tree": GradientBoostingClassifier(random_state=random_state),
    }

    # Helper to get probabilities safely
    def get_proba(model, Xt):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(Xt)[:, 1]
        # fallback (shouldn’t happen here)
        pred = model.predict(Xt)
        return pred.astype(float)

    # Train all models and collect metrics
    rows = []
    cm_train = {}
    cm_test = {}
    roc_train = {}
    roc_test = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Predict
        yhat_tr = model.predict(X_train)
        yhat_te = model.predict(X_test)

        # Probabilities
        p_tr = get_proba(model, X_train)
        p_te = get_proba(model, X_test)

        # Metrics
        train_acc = accuracy_score(y_train, yhat_tr)
        test_acc = accuracy_score(y_test, yhat_te)

        pr_tr = precision_score(y_train, yhat_tr, zero_division=0)
        re_tr = recall_score(y_train, yhat_tr, zero_division=0)
        f1_tr = f1_score(y_train, yhat_tr, zero_division=0)

        pr_te = precision_score(y_test, yhat_te, zero_division=0)
        re_te = recall_score(y_test, yhat_te, zero_division=0)
        f1_te = f1_score(y_test, yhat_te, zero_division=0)

        auc_tr = roc_auc_score(y_train, p_tr)
        auc_te = roc_auc_score(y_test, p_te)

        rows.append([name, train_acc, test_acc, pr_tr, re_tr, f1_tr, pr_te, re_te, f1_te, auc_tr, auc_te])

        cm_train[name] = confusion_matrix(y_train, yhat_tr)
        cm_test[name] = confusion_matrix(y_test, yhat_te)

        roc_train[name] = roc_curve(y_train, p_tr)
        roc_test[name] = roc_curve(y_test, p_te)

    res = pd.DataFrame(rows, columns=[
        "Model", "Train Acc", "Test Acc",
        "Train Precision", "Train Recall", "Train F1",
        "Test Precision", "Test Recall", "Test F1",
        "Train AUC", "Test AUC"
    ])

    # ---------- 1) Train vs Test Accuracy ----------
    st.subheader("1) Train Accuracy vs Test Accuracy")
    fig_acc = px.bar(
        res.melt(id_vars="Model", value_vars=["Train Acc", "Test Acc"], var_name="Split", value_name="Accuracy"),
        x="Model", y="Accuracy", color="Split", barmode="group",
        text_auto=".3f"
    )
    fig_acc.update_layout(height=420, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("**Key insights**")
    st.markdown("- Higher Train accuracy than Test accuracy indicates overfitting risk.")
    st.markdown("- A model with slightly lower accuracy but better generalization is preferred.")

    st.divider()

    # ---------- 2) Confusion Matrix (Train & Test) ----------
    st.subheader("2) Confusion Matrix (Train and Test)")
    model_pick = st.selectbox("Select model for Confusion Matrices", res["Model"].tolist(), index=0)
    mode = st.radio("Display", ["Counts", "Percentages"], horizontal=True)

    def plot_cm(cm, title):
        cm = cm.astype(float)
        z = cm / cm.sum() * 100 if mode == "Percentages" else cm
        txt = np.round(z, 1).astype(str) + "%" if mode == "Percentages" else cm.astype(int).astype(str)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=["Pred: No (0)", "Pred: Yes (1)"],
            y=["Actual: No (0)", "Actual: Yes (1)"],
            text=txt,
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate="Value: %{z}<extra></extra>"
        ))
        fig.update_layout(title=title, height=420, margin=dict(l=40, r=20, t=60, b=40))
        return fig

    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_cm(cm_train[model_pick], f"{model_pick} — Train Confusion Matrix"), use_container_width=True)
    c2.plotly_chart(plot_cm(cm_test[model_pick], f"{model_pick} — Test Confusion Matrix"), use_container_width=True)

    st.divider()

    # ---------- 3) Precision / Recall / F1 (Train & Test) ----------
    st.subheader("3) Precision, Recall, F1-score (Train and Test)")
    metric_df = res[["Model","Train Precision","Train Recall","Train F1","Test Precision","Test Recall","Test F1"]].copy()
    st.dataframe(metric_df, use_container_width=True)

    st.markdown("**Key insights**")
    st.markdown("- Precision: Out of predicted Yes, how many were actually Yes.")
    st.markdown("- Recall: Out of actual Yes, how many were correctly predicted Yes (important for loan targeting).")
    st.markdown("- F1 balances precision and recall; useful for imbalanced acceptance rates.")

    st.divider()

    # ---------- 4) ROC Curves (Train & Test, single chart each) ----------
    st.subheader("4) ROC Curves (Train and Test) — All Models")

    color_map = {
        "KNN": "#1f77b4",
        "Decision Tree": "#ff7f0e",
        "Random Forest": "#2ca02c",
        "Gradient Boosted Tree": "#d62728",
    }

    def plot_roc_all(roc_dict, auc_col, title):
        fig = go.Figure()
        for name in res["Model"]:
            fpr, tpr, _ = roc_dict[name]
            auc_val = float(res.loc[res["Model"] == name, auc_col].iloc[0])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{name} (AUC={auc_val:.3f})",
                line=dict(color=color_map.get(name, None), width=3)
            ))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=520,
            legend=dict(orientation="v")
        )
        return fig

    left, right = st.columns([2.2, 1])
    left.plotly_chart(plot_roc_all(roc_train, "Train AUC", "ROC Curve — Training Data"), use_container_width=True)
    right.dataframe(res[["Model","Train AUC"]].sort_values("Train AUC", ascending=False), use_container_width=True)

    left2, right2 = st.columns([2.2, 1])
    left2.plotly_chart(plot_roc_all(roc_test, "Test AUC", "ROC Curve — Testing Data"), use_container_width=True)
    right2.dataframe(res[["Model","Test AUC"]].sort_values("Test AUC", ascending=False), use_container_width=True)

    st.markdown("**Key insights**")
    st.markdown("- ROC curve closer to top-left indicates better discrimination.")
    st.markdown("- Higher AUC means better ability to separate Yes vs No.")
    st.markdown("- Compare Train vs Test ROC to detect overfitting.")

    st.divider()

    # Extra requested check: CCAvg > 0 but CreditCard = 0 -> Mortgage / Loan
    st.subheader("📌 Extra Check: CCAvg present but No Credit Card")
    if COL_CCAVG and COL_CREDITCARD:
        odd = dff[(dff[COL_CCAVG] > 0) & (dff[COL_CREDITCARD] == 0)].copy()
        st.write(f"Count: **{len(odd):,}**")

        if len(odd) > 0:
            loan_rate_odd = odd[COL_TARGET].mean()
            mort_rate_odd = (odd[COL_MORTGAGE] > 0).mean() if COL_MORTGAGE else np.nan

            colx, coly = st.columns(2)
            colx.metric("Personal Loan Yes %", f"{loan_rate_odd*100:.1f}%")
            coly.metric("Mortgage > 0 %", f"{mort_rate_odd*100:.1f}%" if COL_MORTGAGE else "NA")

            key_insights([
                "If many such customers have mortgage, spending may be estimated from banking relationship rather than explicit card ownership.",
                "If many accepted personal loan, this segment may represent strong banking engagement despite CreditCard=0."
            ])

            st.dataframe(odd[[COL_AGE, COL_INCOME, COL_CCAVG, COL_MORTGAGE, COL_TARGET]].head(25), use_container_width=True)
    else:
        st.info("CCAvg or CreditCard column missing — cannot run this check.")

