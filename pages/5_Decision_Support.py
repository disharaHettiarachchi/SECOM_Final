"""Streamlit page for operational decision-support outputs."""

import pandas as pd
import streamlit as st

from src.data_loader import get_feature_columns, load_secom_dataset
from src.model_training import load_model_artifact, load_uploaded_model_artifact
from src.prediction import high_risk_records, quality_recommendations
from src.visualizations import plot_risk_records


st.set_page_config(page_title="Decision Support", layout="wide")


@st.cache_data(show_spinner=False)
def cached_dataset():
    return load_secom_dataset()


def get_or_load_artifact():
    if "model_artifact" in st.session_state:
        return st.session_state.model_artifact
    artifact = load_model_artifact()
    if artifact:
        st.session_state.model_artifact = artifact
    return artifact


st.title("Decision Support")

data = cached_dataset()
feature_columns = get_feature_columns(data)
artifact = get_or_load_artifact()

if artifact is None:
    st.warning("A trained model artifact is required for high-risk record ranking.")
    uploaded_artifact = st.file_uploader(
        "Upload `secom_fault_detection_model.joblib` exported from Colab",
        type=["joblib"],
    )
    if uploaded_artifact is not None:
        try:
            artifact = load_uploaded_model_artifact(uploaded_artifact)
            st.session_state.model_artifact = artifact
            st.success(f"Uploaded model loaded: {artifact['model_name']}")
        except Exception as exc:
            st.error(f"Could not load the uploaded model: {exc}")
            st.stop()
    if artifact is None:
        st.info(
            "Run the Colab training notebook and place the exported model artifact "
            "in `models/` before deploying, or upload it here during a demo."
        )
        st.stop()

st.success(f"Active model: {artifact['model_name']}")

control_1, control_2 = st.columns(2)
with control_1:
    top_n = st.slider("High-risk records to display", 5, 50, 20, 5)
with control_2:
    threshold = st.slider(
        "Fault decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(artifact.get("decision_threshold", 0.50)),
        step=0.05,
        help="Lower values increase sensitivity and the number of records flagged.",
    )

try:
    all_ranked_records = high_risk_records(
        artifact,
        data,
        feature_columns,
        top_n=len(data),
        threshold=threshold,
    )
except AttributeError as exc:
    st.error(
        "The saved model is incompatible with the installed scikit-learn version. "
        "Install the pinned requirements.txt file (scikit-learn 1.6.1) and redeploy. "
        f"Technical detail: {exc}"
    )
    st.stop()
except Exception as exc:
    st.error(f"High-risk screening could not be completed: {exc}")
    st.stop()

flagged_records = all_ranked_records[
    all_ranked_records["fault_probability"] >= threshold
].copy()

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Records screened", f"{len(data):,}")
metric_2.metric("Flagged for review", f"{len(flagged_records):,}")
metric_3.metric(
    "Known faults in flagged set",
    f"{int(flagged_records['target'].sum()):,}" if not flagged_records.empty else "0",
)

risk_filter = st.multiselect(
    "Risk bands to include",
    ["High risk", "Medium risk", "Low risk"],
    default=["High risk", "Medium risk"],
)
ranked_records = all_ranked_records[
    all_ranked_records["risk_band"].isin(risk_filter)
].head(top_n)

st.subheader("Highest-Risk Production Records")
if ranked_records.empty:
    st.info("No records match the selected risk bands and threshold settings.")
    st.stop()

st.plotly_chart(plot_risk_records(ranked_records), use_container_width=True)
st.dataframe(
    ranked_records,
    use_container_width=True,
    hide_index=True,
    column_config={
        "timestamp": "Timestamp",
        "condition": "Actual condition",
        "target": "Actual target",
        "predicted_condition": "Predicted condition",
        "fault_probability": st.column_config.ProgressColumn(
            "Fault probability",
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        ),
        "risk_band": "Risk band",
    },
)

st.download_button(
    "Download ranked risk register",
    data=ranked_records.to_csv(index=False).encode("utf-8"),
    file_name="secom_high_risk_records.csv",
    mime="text/csv",
)

st.subheader("Action Matrix")
action_rows = []
for band in ["High risk", "Medium risk", "Low risk"]:
    action_rows.append(
        {
            "risk_band": band,
            "recommended_actions": " | ".join(quality_recommendations(band)),
        }
    )

st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

st.warning(
    "Predictions should support, not replace, engineering judgement. A high-risk "
    "score means the production record should be prioritised for inspection and "
    "process review."
)

st.caption(
    "The labels shown here come from the historical SECOM dataset and are included "
    "for retrospective evaluation. In a live workflow, the true condition would only "
    "become available after inspection or downstream testing."
)
