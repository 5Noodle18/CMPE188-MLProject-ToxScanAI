import streamlit as st
import torch
import pandas as pd

from src.CONDA import get_device, load_model, predict_proba, LABEL_NAMES

EXPLANATION_RULES = {
    "Explicit": ["idiot", "trash", "stupid", "loser", "moron", "dumb"],
    "Implicit": ["nice job", "sure buddy", "wow amazing", "good one"],
    "Action": ["reported", "muted", "kick", "ban"],
    "Other": []
}

def explain_prediction(text, label):
    text_lower = text.lower()
    reasons = []

    for keyword in EXPLANATION_RULES.get(label, []):
        if keyword in text_lower:
            reasons.append(f'Contains phrase: "{keyword}"')

    if label == "Explicit":
        reasons.append("Direct hostile language detected")
    elif label == "Implicit":
        reasons.append("Possible sarcasm or passive-aggressive tone")
    elif label == "Action":
        reasons.append("References moderation or player actions")
    else:
        reasons.append("No strong toxic language detected")

    return reasons

st.set_page_config(page_title="ToxScan AI", page_icon="🛡️")

@st.cache_resource
def load_toxscan():
    device = get_device()
    model = load_model(device, output_dir="output", prefix="toxscan")
    if model is None:
        st.error("No trained model found. Run: python src/CONDA.py first")
        st.stop()
    return model, device

model, device = load_toxscan()

st.title("🛡️ ToxScan AI")
st.write("Enter gaming chat text and get a toxicity classification.")

text = st.text_area("Chat message", placeholder="you are so trash get out of this game")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        probs = predict_proba(model, [text], device)[0]
        pred_idx = int(probs.argmax())
        pred_label = LABEL_NAMES[pred_idx]
        confidence = probs[pred_idx]

        st.subheader("Result")
        st.metric("Prediction", pred_label)
        st.metric("Confidence", f"{confidence:.2%}")

        reasons = explain_prediction(text, pred_label)

        st.subheader("Why this prediction?")
        for r in reasons:
            st.write(f"- {r}")

        df = pd.DataFrame({
            "Class": LABEL_NAMES,
            "Probability": probs
        })

        st.bar_chart(df.set_index("Class"))
        st.dataframe(df)