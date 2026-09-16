

import re
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

# ----------------------------- Page setup -----------------------------
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="centered")

# Minimal styling only — no forced background, so it follows the user's
# light/dark theme setting automatically.
st.markdown("""
<style>
.result-card {
    border-left: 6px solid var(--accent-color);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-top: 0.4rem;
    background: rgba(127, 127, 127, 0.08);
}
.result-card h4 { margin: 0 0 0.3rem 0; font-size: 0.9rem; opacity: 0.7; }
.result-card .label { font-size: 1.3rem; font-weight: 700; }
.result-card .conf { font-size: 0.85rem; opacity: 0.7; }
.stButton>button {
    background: #0ea5a4;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

SENTIMENT_STYLE = {
    "Positive":   {"color": "#22c55e", "emoji": "😊"},
    "Negative":   {"color": "#f43f5e", "emoji": "😠"},
    "Neutral":    {"color": "#0ea5e9", "emoji": "😐"},
    "Irrelevant": {"color": "#a855f7", "emoji": "🤷"},
}

LABELS = ["Irrelevant", "Negative", "Neutral", "Positive"]  # alphabetical, matches LabelEncoder

# ----------------------------- Text cleaning (same as training) -----------------------------
# NOTE: the notebook keeps "!" and "?" in the cleaned text — they're used as a
# sentiment signal — so this must match exactly for all four models
# (SVC, SimpleRNN, LSTM, GRU) to get correctly-preprocessed input.
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s!?]', ' ', text)   # keep ! and ? — sentiment signal
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ----------------------------- Load models (cached) -----------------------------
@st.cache_resource
def load_svm():
    with open("sentiment_svm_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return pipeline, le

@st.cache_resource
def load_rnn():
    return tf.keras.models.load_model("sentiment_rnn_model.keras")

@st.cache_resource
def load_lstm():
    return tf.keras.models.load_model("LSTM_model.keras")

@st.cache_resource
def load_gru():
    return tf.keras.models.load_model("GRU_model.keras")

def svm_predict(pipeline, le, clean_text):
    scores = pipeline.decision_function([clean_text])[0]
    probs = np.exp(scores) / np.exp(scores).sum()
    pred_idx = int(np.argmax(probs))
    return le.classes_[pred_idx], float(probs[pred_idx])

def keras_predict(model, clean_text):
    """Shared prediction path for SimpleRNN / LSTM / GRU — they all take raw
    text directly since TextVectorization is baked into the model."""
    probs = model.predict(tf.constant([clean_text]), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], float(probs[pred_idx])

def render_result(model_name, label, confidence):
    style = SENTIMENT_STYLE.get(label, {"color": "#888", "emoji": ""})
    st.markdown(f"""
    <div class="result-card" style="--accent-color:{style['color']}">
        <h4>{model_name}</h4>
        <div class="label" style="color:{style['color']}">{style['emoji']} {label}</div>
        <div class="conf">Confidence: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------- UI -----------------------------
st.title("💬 Tweet Sentiment Analyzer")
st.caption("SVC (TF-IDF + LinearSVC) vs. SimpleRNN vs. LSTM vs. GRU")

MODEL_LOADERS = {
    "SVC": load_svm,
    "SimpleRNN": load_rnn,
    "LSTM": load_lstm,
    "GRU": load_gru,
}
MODEL_FILES = {
    "SVC": "sentiment_svm_pipeline.pkl / label_encoder.pkl",
    "SimpleRNN": "sentiment_rnn_model.keras",
    "LSTM": "LSTM_model.keras",
    "GRU": "GRU_model.keras",
}

model_choice = st.selectbox("Choose a model:", ["SVC", "SimpleRNN", "LSTM", "GRU"])
tweet = st.text_area("Enter a tweet:", height=110, placeholder="e.g. This new update completely ruined the game...")
run = st.button("Analyze")

if run:
    if not tweet.strip():
        st.warning("Please type a tweet first.")
    else:
        cleaned = clean_tweet(tweet)

        try:
            if model_choice == "SVC":
                pipeline, le = load_svm()
                label, conf = svm_predict(pipeline, le, cleaned)
            else:
                model = MODEL_LOADERS[model_choice]()
                label, conf = keras_predict(model, cleaned)
            render_result(model_choice, label, conf)
        except (FileNotFoundError, OSError):
            st.error(f"Missing {MODEL_FILES[model_choice]}")