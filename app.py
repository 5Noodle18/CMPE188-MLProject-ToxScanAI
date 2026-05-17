import streamlit as st
import pandas as pd

from src.CONDA import (
    get_device,
    load_model,
    predict_proba,
    LABEL_NAMES,
    explain_prediction
)

st.set_page_config(
    page_title="ToxScan AI",
    page_icon="🛡️"
)


@st.cache_resource
def load_toxscan():
    device = get_device()

    model = load_model(
        device,
        output_dir="output",
        prefix="toxscan"
    )

    if model is None:
        st.error("No trained model found. Run: python src/CONDA.py first")
        st.stop()

    return model, device


model, device = load_toxscan()

st.title("🛡️ ToxScan AI")
st.write("ToxScan analyzes gaming chat messages for toxicity. Enter in a chat message.")

text = st.text_area(
    "Chat message",
    placeholder="you are so trash get out of this game"
)

if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter a message.")

    else:
        probs = predict_proba(model, [text], device)[0]

        pred_idx = int(probs.argmax())
        pred_label = LABEL_NAMES[pred_idx]
        confidence = probs[pred_idx]

        # Confidence threshold
        if confidence < 0.60:
            pred_label = "Uncertain"

        st.subheader("Result")

        st.metric("Prediction", pred_label)
        st.metric("Confidence", f"{confidence:.2%}")

        reasons = explain_prediction(text, pred_label)

        st.subheader("Why this prediction?")

        for reason in reasons:
            st.write(f"- {reason}")

        df = pd.DataFrame({
            "Class": LABEL_NAMES,
            "Probability": probs
        })

        st.subheader("Class Probabilities")

        st.bar_chart(df.set_index("Class"))
        st.dataframe(df)