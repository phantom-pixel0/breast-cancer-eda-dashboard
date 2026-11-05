from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer | Patient Analytics",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#14243A"
NAVY_MID = "#213A56"
PINK = "#E89AB4"
PINK_DARK = "#C96F91"
LIGHT = "#F4F6F9"
INK = "#26364A"
MUTED = "#718096"
PALETTE = [PINK_DARK, "#6C9BB8", "#E9BE83", "#8EB5A6", "#9C8BC5", "#D89E83"]


st.markdown(
    f"""
    <style>
      :root {{ color-scheme: light; }}
      .stApp {{ background: {LIGHT}; color: {INK}; }}
      [data-testid="stHeader"] {{ background: rgba(244,246,249,.92); }}
      [data-testid="stSidebar"] {{ background: {NAVY}; }}
      [data-testid="stSidebar"] * {{ color: #F7F9FC; }}
      [data-testid="stSidebar"] [data-baseweb="select"] div {{ color: {INK}; }}
      [data-testid="stSidebar"] [data-baseweb="tag"] {{ background: {PINK_DARK}; }}
      h1, h2, h3 {{ color: {NAVY}; letter-spacing: -.025em; }}
      .hero {{ background: linear-gradient(115deg,{NAVY},{NAVY_MID}); padding: 1.65rem 1.9rem;
        border-radius: 20px; margin: .3rem 0 1.35rem; box-shadow: 0 10px 28px rgba(20,36,58,.14); }}
      .hero h1 {{ color: white; margin: 0; font-size: 2rem; }}
      .hero p {{ color: #D6E0EB; margin: .35rem 0 0; font-size: 1rem; }}
      .eyebrow {{ color: {PINK}; text-transform: uppercase; letter-spacing: .12em; font-size: .75rem;
        font-weight: 700; margin-bottom: .45rem; }}
      div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #E4E9F0; padding: 1rem 1.1rem;
        border-radius: 16px; box-shadow: 0 5px 16px rgba(20,36,58,.055); }}
      div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
      div[data-testid="stMetricValue"] {{ color: {NAVY}; }}
      .insight {{ background: white; border: 1px solid #E4E9F0; border-left: 4px solid {PINK};
        border-radius: 12px; padding: .9rem 1rem; margin: .55rem 0; box-shadow: 0 4px 12px rgba(20,36,58,.04); }}
      .section-note {{ color: {MUTED}; margin-top: -.35rem; margin-bottom: 1.1rem; }}
      [data-testid="stPlotlyChart"] {{ background: white; border: 1px solid #E7EBF1; border-radius: 16px;
        padding: .4rem; box-shadow: 0 5px 16px rgba(20,36,58,.045); }}
      .block-container {{ padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1500px; }}
      footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load the source file, normalize its headers and derive EDA categories."""
    csv_path = Path(__file__).resolve().parent / "Breast_Cancer_Cleaned.csv"

    data = pd.read_csv(csv_path)

    data.columns = (
        data.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    data = data.drop_duplicates().copy()
    for col in [
        "age", "tumor_size", "survival_months", "regional_node_examined",
        "reginol_node_positive",
    ]:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype("string").str.strip()
    if "age" in data:
        data["age_group"] = pd.cut(
            data["age"], [0, 40, 60, np.inf],
            labels=["Young Adult", "Middle-Aged", "Older Adult"], include_lowest=True,
        )
    if "tumor_size" in data:
        data["tumor_size_category"] = pd.cut(
            data["tumor_size"], [0, 20, 50, np.inf],
            labels=["Small", "Medium", "Large"], include_lowest=True,
        )
    return data


def has(data, *columns):
    return all(column in data.columns for column in columns)


def title_block(title, description):
    st.markdown(f"## {title}")
    st.markdown(f'<div class="section-note">{description}</div>', unsafe_allow_html=True)


def chart_layout(fig, height=390):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": INK, "size": 12},
        title={"font": {"size": 16, "color": NAVY}, "x": 0.03, "xanchor": "left"},
        margin={"l": 28, "r": 20, "t": 58, "b": 35},
        legend={"orientation": "h", "y": 1.06, "x": 0.02},
        hoverlabel={"bgcolor": "white", "font_color": NAVY},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title_font_color=MUTED)
    fig.update_yaxes(gridcolor="#E9EDF2", zeroline=False, title_font_color=MUTED)
    return fig


def show_chart(fig, key):
    st.plotly_chart(chart_layout(fig), width="stretch", key=key)


def counts_chart(data, column, title, key, horizontal=False):
    if not has(data, column):
        return
    series = data[column].astype("string").fillna("Missing")
    summary = series.value_counts(dropna=False).rename_axis(column).reset_index(name="Patients")
    if isinstance(data[column].dtype, pd.CategoricalDtype):
        order = [str(value) for value in data[column].cat.categories if (series == str(value)).any()]
        summary[column] = pd.Categorical(summary[column], categories=order, ordered=True)
        summary = summary.sort_values(column)
    if horizontal:
        fig = px.bar(summary, x="Patients", y=column, orientation="h", title=title,
                     color="Patients", color_continuous_scale=["#F4D9E2", PINK_DARK], text="Patients")
    else:
        fig = px.bar(summary, x=column, y="Patients", title=title, color=column,
                     color_discrete_sequence=PALETTE, text="Patients")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    show_chart(fig, key)


def numeric_hist(data, column, title, key, color=PINK_DARK):
    valid = data[column].dropna() if column in data else pd.Series(dtype=float)
    if valid.empty:
        st.info(f"No valid values available for {column.replace('_', ' ')}.")
        return
    fig = px.histogram(data, x=column, nbins=30, marginal="box", title=title,
                       color_discrete_sequence=[color])
    fig.update_traces(hovertemplate=f"{column.replace('_', ' ').title()}: %{{x}}<br>Patients: %{{y}}<extra></extra>")
    show_chart(fig, key)


def status_bar(data, group, title, key):
    if not has(data, group, "status"):
        return
    clean = data.dropna(subset=[group, "status"])
    if clean.empty:
        return
    summary = clean.groupby([group, "status"], observed=True).size().reset_index(name="Patients")
    fig = px.bar(summary, x=group, y="Patients", color="status", barmode="group", title=title,
                 color_discrete_map={"Alive": "#6C9BB8", "Dead": PINK_DARK})
    show_chart(fig, key)


def survival_box(data, category, title, key):
    if not has(data, category, "survival_months"):
        return
    clean = data.dropna(subset=[category, "survival_months"])
    if clean.empty:
        return
    fig = px.box(clean, x=category, y="survival_months", color=category, points="outliers",
                 title=title, color_discrete_sequence=PALETTE)
    show_chart(fig, key)


def add_insight(message):
    st.markdown(f'<div class="insight">{message}</div>', unsafe_allow_html=True)


try:
    df = load_data()
except FileNotFoundError:
    st.error("Breast_Cancer_Cleaned.csv was not found beside dashboard.py.")
    st.stop()
except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if df.empty:
    st.warning("The dataset contains no records.")
    st.stop()

st.sidebar.markdown("# 🎗️ Patient Analytics")
st.sidebar.caption("Breast cancer dataset · descriptive exploration")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "DASHBOARD", ["Overview", "Patient Demographics", "Cancer Characteristics", "Survival Analysis", "Data Insights"],
    label_visibility="visible",
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Filter records")
filtered = df.copy()

if has(filtered, "age_group"):
    age_opts = [str(v) for v in filtered["age_group"].dropna().unique()]
    selected = st.sidebar.multiselect("Age group", age_opts, default=age_opts)
    filtered = filtered[filtered["age_group"].astype("string").isin(selected)]
if has(filtered, "race"):
    opts = sorted(filtered["race"].dropna().astype(str).unique())
    selected = st.sidebar.multiselect("Race", opts, default=opts)
    filtered = filtered[filtered["race"].astype("string").isin(selected)]
if has(filtered, "6th_stage"):
    opts = sorted(filtered["6th_stage"].dropna().astype(str).unique())
    selected = st.sidebar.multiselect("Cancer stage", opts, default=opts)
    filtered = filtered[filtered["6th_stage"].astype("string").isin(selected)]
if has(filtered, "status"):
    opts = sorted(filtered["status"].dropna().astype(str).unique())
    selected = st.sidebar.multiselect("Patient status", opts, default=opts)
    filtered = filtered[filtered["status"].astype("string").isin(selected)]

st.sidebar.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** unique records")
st.sidebar.caption("Age groups follow the EDA bins: ≤40, 41–60, and >60 years.")

st.markdown(
    '<div class="hero"><div class="eyebrow">Healthcare data analytics</div>'
    '<h1>Breast Cancer Patient Dashboard</h1>'
    '<p>Explore patient demographics, tumor characteristics and recorded survival duration.</p></div>',
    unsafe_allow_html=True,
)
st.caption("Educational, descriptive analysis only. Recorded survival months describe this dataset and are not individual predictions.")

if filtered.empty:
    st.warning("No records match the selected filters. Adjust the sidebar filters to continue.")
    st.stop()


if page == "Overview":
    title_block("Overview", "A filtered snapshot of the records, patient status and major cohort distributions.")
    alive = int(filtered["status"].astype("string").str.casefold().eq("alive").sum()) if "status" in filtered else 0
    dead = int(filtered["status"].astype("string").str.casefold().eq("dead").sum()) if "status" in filtered else 0
    alive_pct = alive / len(filtered) * 100 if len(filtered) else np.nan
    metrics = st.columns(5)
    metrics[0].metric("Patients", f"{len(filtered):,}")
    metrics[1].metric("Average age", f"{filtered['age'].mean():.1f} yr" if has(filtered, "age") and filtered["age"].notna().any() else "N/A")
    metrics[2].metric("Average tumor size", f"{filtered['tumor_size'].mean():.1f}" if has(filtered, "tumor_size") and filtered["tumor_size"].notna().any() else "N/A")
    metrics[3].metric("Alive", f"{alive:,}", f"{alive_pct:.1f}% of filtered records" if pd.notna(alive_pct) and "status" in filtered else None)
    metrics[4].metric("Deceased", f"{dead:,}")
    left, right = st.columns(2)
    with left:
        if "status" in filtered:
            status_counts = filtered["status"].fillna("Missing").value_counts().rename_axis("Status").reset_index(name="Patients")
            fig = px.pie(status_counts, names="Status", values="Patients", hole=.62, title="Patient status", color="Status",
                         color_discrete_map={"Alive": "#6C9BB8", "Dead": PINK_DARK})
            show_chart(fig, "overview_status")
        if "age" in filtered:
            numeric_hist(filtered, "age", "Age distribution", "overview_age")
    with right:
        counts_chart(filtered, "6th_stage", "Patients by cancer stage", "overview_stage")
        counts_chart(filtered, "tumor_size_category", "Tumor size categories", "overview_tumor_category")
    if has(filtered, "survival_months"):
        cols = st.columns(3)
        cols[0].metric("Mean recorded survival", f"{filtered['survival_months'].mean():.1f} months")
        cols[1].metric("Median recorded survival", f"{filtered['survival_months'].median():.1f} months")
        cols[2].metric("Recorded range", f"{filtered['survival_months'].min():.0f}–{filtered['survival_months'].max():.0f} months")

elif page == "Patient Demographics":
    title_block("Patient Demographics", "Explore the age profile and the recorded race and marital-status categories.")
    left, right = st.columns(2)
    with left:
        numeric_hist(filtered, "age", "Age distribution", "demo_age")
        counts_chart(filtered, "race", "Patients by race", "demo_race", horizontal=True)
    with right:
        counts_chart(filtered, "age_group", "Patients by age group", "demo_age_group")
        counts_chart(filtered, "marital_status", "Marital status distribution", "demo_marital", horizontal=True)
    status_bar(filtered, "age_group", "Patient status by age group", "demo_status_age")

elif page == "Cancer Characteristics":
    title_block("Cancer Characteristics", "Tumor size, stage, receptor status and regional lymph-node measurements in the selected cohort.")
    left, right = st.columns(2)
    with left:
        numeric_hist(filtered, "tumor_size", "Tumor size distribution", "cancer_tumor")
        counts_chart(filtered, "6th_stage", "Overall cancer stage", "cancer_overall_stage")
        counts_chart(filtered, "t_stage", "T stage", "cancer_tstage")
        counts_chart(filtered, "estrogen_status", "Estrogen receptor status", "cancer_estrogen")
        numeric_hist(filtered, "regional_node_examined", "Regional nodes examined", "cancer_nodes_examined", color="#6C9BB8")
    with right:
        counts_chart(filtered, "tumor_size_category", "Tumor size category", "cancer_tumor_cat")
        counts_chart(filtered, "n_stage", "N stage", "cancer_nstage")
        counts_chart(filtered, "progesterone_status", "Progesterone receptor status", "cancer_progesterone")
        numeric_hist(filtered, "reginol_node_positive", "Positive regional nodes", "cancer_nodes_positive", color="#6C9BB8")
    status_bar(filtered, "t_stage", "Patient status by T stage", "cancer_status_t")
    if has(filtered, "reginol_node_positive", "status"):
        fig = px.box(filtered.dropna(subset=["reginol_node_positive", "status"]), x="status", y="reginol_node_positive",
                     color="status", title="Positive regional nodes by patient status",
                     color_discrete_map={"Alive": "#6C9BB8", "Dead": PINK_DARK}, points="outliers")
        show_chart(fig, "cancer_nodes_status")

elif page == "Survival Analysis":
    title_block("Survival Analysis", "Compare recorded survival months across patient status, age groups and tumor characteristics.")
    if "survival_months" in filtered:
        cols = st.columns(3)
        cols[0].metric("Mean recorded months", f"{filtered['survival_months'].mean():.1f}")
        cols[1].metric("Median recorded months", f"{filtered['survival_months'].median():.1f}")
        cols[2].metric("Maximum recorded months", f"{filtered['survival_months'].max():.0f}")
        numeric_hist(filtered, "survival_months", "Recorded survival months distribution", "survival_hist", color="#6C9BB8")
        left, right = st.columns(2)
        with left:
            survival_box(filtered, "status", "Recorded months by patient status", "survival_status_box")
        with right:
            survival_box(filtered, "age_group", "Recorded months by age group", "survival_age_box")
        survival_box(filtered, "6th_stage", "Recorded months by cancer stage", "survival_stage_box")
        if has(filtered, "tumor_size", "survival_months"):
            fields = ["tumor_size", "survival_months"] + (["status"] if "status" in filtered else [])
            plot = filtered[fields].dropna(subset=["tumor_size", "survival_months"])
            fig = px.scatter(plot, x="tumor_size", y="survival_months", color="status" if "status" in plot else None,
                             color_discrete_map={"Alive": "#6C9BB8", "Dead": PINK_DARK}, opacity=.65,
                             title="Tumor size vs recorded survival months", hover_data=fields)
            show_chart(fig, "survival_tumor_scatter")
        if has(filtered, "reginol_node_positive", "survival_months"):
            plot = filtered.dropna(subset=["reginol_node_positive", "survival_months"])
            fig = px.scatter(plot, x="reginol_node_positive", y="survival_months", color="status" if "status" in plot else None,
                             color_discrete_map={"Alive": "#6C9BB8", "Dead": PINK_DARK}, opacity=.65,
                             title="Positive regional nodes vs recorded survival months")
            show_chart(fig, "survival_nodes_scatter")
    else:
        st.info("The source data does not include a survival_months column.")

elif page == "Data Insights":
    title_block("Data Insights", "Findings are recalculated from the filtered source records. They describe associations, not causation.")
    st.markdown("### Cohort summary")
    st.write(f"**{len(filtered):,}** records are currently selected from **{len(df):,}** unique source records. "
             f"The filtered data contains **{int(filtered.isna().sum().sum()):,}** missing cells.")

    if has(filtered, "age_group", "status"):
        age_status = pd.crosstab(filtered["age_group"], filtered["status"], normalize="index") * 100
        alive_col = next((c for c in age_status.columns if str(c).casefold() == "alive"), None)
        dead_col = next((c for c in age_status.columns if str(c).casefold() == "dead"), None)
        if alive_col is not None and not age_status.empty:
            top_age = age_status[alive_col].idxmax()
            low_age = age_status[alive_col].idxmin()
            add_insight(f"The highest recorded Alive share by age group is <b>{top_age}</b> ({age_status.loc[top_age, alive_col]:.1f}%); the lowest is <b>{low_age}</b> ({age_status.loc[low_age, alive_col]:.1f}%).")
        if dead_col is not None and not age_status.empty:
            top_dead = age_status[dead_col].idxmax()
            add_insight(f"The highest recorded Dead share by age group is <b>{top_dead}</b> ({age_status.loc[top_dead, dead_col]:.1f}%).")

    if has(filtered, "status", "tumor_size"):
        means = filtered.groupby("status", observed=True)["tumor_size"].mean().dropna()
        if len(means) >= 2:
            a, b = means.get("Alive", np.nan), means.get("Dead", np.nan)
            if pd.notna(a) and pd.notna(b):
                add_insight(f"Mean recorded tumor size is <b>{b:.2f}</b> for Dead records and <b>{a:.2f}</b> for Alive records.")
    if has(filtered, "status", "reginol_node_positive"):
        means = filtered.groupby("status", observed=True)["reginol_node_positive"].mean().dropna()
        if "Alive" in means and "Dead" in means:
            add_insight(f"Mean positive regional nodes: <b>{means['Dead']:.2f}</b> for Dead records and <b>{means['Alive']:.2f}</b> for Alive records.")
    if has(filtered, "t_stage", "status"):
        stage_dead = pd.crosstab(filtered["t_stage"], filtered["status"], normalize="index") * 100
        dead_col = next((c for c in stage_dead.columns if str(c).casefold() == "dead"), None)
        if dead_col is not None and not stage_dead.empty:
            stage = stage_dead[dead_col].idxmax()
            add_insight(f"Among recorded T stages, <b>{stage}</b> has the highest Dead share ({stage_dead.loc[stage, dead_col]:.1f}%).")
    if has(filtered, "estrogen_status", "progesterone_status", "status"):
        receptors = filtered.groupby(["estrogen_status", "progesterone_status", "status"], observed=True).size()
        if not receptors.empty:
            combo, count = receptors.idxmax(), int(receptors.max())
            add_insight(f"The most frequent receptor/status combination is <b>{combo[0]} estrogen, {combo[1]} progesterone, {combo[2]}</b> ({count:,} records).")
    corr_columns = [c for c in ["regional_node_examined", "reginol_node_positive", "tumor_size", "survival_months"] if c in filtered]
    if len(corr_columns) >= 2:
        corr = filtered[corr_columns].corr(numeric_only=True)
        pairs = [("regional_node_examined", "reginol_node_positive"), ("tumor_size", "reginol_node_positive"), ("survival_months", "reginol_node_positive")]
        lines = []
        for a, b in pairs:
            if a in corr and b in corr and pd.notna(corr.loc[a, b]):
                lines.append(f"{a.replace('_', ' ')} with {b.replace('_', ' ')}: <b>{corr.loc[a, b]:.2f}</b>")
        if lines:
            add_insight("Pearson correlations in the selected records — " + "; ".join(lines) + ".")
    st.markdown("### EDA reference findings")
    st.caption("The project notebook reports the following full-dataset findings. These refer to the original cohort and do not change with sidebar filters.")
    add_insight("In the notebook's full-data analysis, Middle-Aged patients had the highest Alive share (87.14%); Older Adults had the highest Dead share (20.11%).")
    add_insight("The notebook reports mean tumor size of 37.14 for Dead patients and 29.27 for Alive patients; median sizes were 30 and 23, respectively.")
    add_insight("The notebook reports mean positive lymph-node counts of 7.24 for Dead patients and 3.60 for Alive patients (medians 4 and 2).")
    add_insight("The notebook reports Dead shares of 9.80% for T1 and 39.22% for T4. Regional nodes examined and positive nodes had correlation 0.41; tumor size and positive nodes 0.24; survival months and positive nodes −0.14.")
    st.caption("Dataset fields and notebook findings are descriptive. They do not establish causation or provide clinical guidance.")

st.markdown("---")
st.caption("Breast Cancer Patient Analytics · Built with Streamlit, Pandas and Plotly · Source: Breast_Cancer.csv")