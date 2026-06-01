from __future__ import annotations

import pandas as pd
import streamlit as st

from src.inference import InjuryRiskPredictor


st.set_page_config(page_title="Athlete Injury Risk", layout="wide")
st.title("Athlete Injury Risk Monitor")

uploaded = st.file_uploader("Upload athlete metrics CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    predictor = InjuryRiskPredictor()
    predictions = predictor.predict_batch(df.to_dict(orient="records"))
    result = pd.concat([df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    st.dataframe(result, use_container_width=True)
else:
    st.info("Upload a CSV with athlete metrics after training the model.")
