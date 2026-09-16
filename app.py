
import re
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

# ----------------------------- Page setup -----------------------------
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg, #0f1220 0%, #1a1f3a 100%);
    color: #eaeaf5;
}
.hero {
    text-align: center;
    padding: 1.6rem 0 0.8rem 0;
}
.hero h1 {
    font-size: 2.4rem;
    background: linear-gradient(90deg, #7f5af0, #2cb67d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero p { color: #a0a0b8; font-size: 1rem; }

.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem;
    text-align: center;
}
.badge {
    display: inline-block;
    padding: 0.5rem 1.4rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.2rem;
    margin: 0.6rem 0;
}
.stButton>button {
    background: linear-gradient(90deg, #7f5af0, #6246ea);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    width: 100%;
}
textarea { background: rgba(255,255,255,0.05) !important; color: #eaeaf5 !important; }
</style>
""", unsafe_allow_html=True)

SENTIMENT_STYLE = {
    "Positive":   {"color": "#2cb67d", "emoji": "😊"},
    "Negative":   {"color": "#e53170", "emoji": "😠"},
    "Neutral":    {"color": "#7f9cf5", "emoji": "😐"},
    "Irrelevant": {"color": "#a0a0b8", "emoji": "🤷"},
}

# ----------------------------- Text cleaning (same as training) -----------------------------
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
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

LABELS = ["Irrelevant", "Negative", "Neutral", "Positive"]  # alphabetical, matches LabelEncoder

def svm_predict(pipeline, le, clean_text):
    scores = pipeline.decision_function([clean_text])[0]
    probs = np.exp(scores) / np.exp(scores).sum()          # softmax over decision scores
    pred_idx = int(np.argmax(probs))
    return le.classes_[pred_idx], probs, list(le.classes_)

def rnn_predict(model, clean_text):
    probs = model.predict(tf.constant([clean_text]), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], probs, LABELS

def render_result(model_name, label, probs, classes):
    style = SENTIMENT_STYLE.get(label, {"color": "#ffffff", "emoji": ""})
    st.markdown(f"""
    <div class="card">
        <p style="color:#a0a0b8;margin-bottom:0;">{model_name}</p>
        <div class="badge" style="background:{style['color']}22;color:{style['color']};border:1px solid {style['color']};">
            {style['emoji']} {label}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.bar_chart({classes[i]: float(probs[i]) for i in range(len(classes))})

# ----------------------------- Header -----------------------------
st.markdown("""
<div class="hero">
    <h1>💬 Tweet Sentiment Analyzer</h1>
    <p>Compare an SVC (TF-IDF + LinearSVC) model against a Bidirectional SimpleRNN</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Sidebar -----------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio("Choose model(s)", ["Both (compare)", "SVC only", "SimpleRNN only"])
st.sidebar.markdown("---")
st.sidebar.caption("Classes: Irrelevant · Negative · Neutral · Positive")

# ----------------------------- Input -----------------------------
tweet = st.text_area("Enter a tweet:", height=110, placeholder="e.g. This new update completely ruined the game...")
run = st.button("Analyze Sentiment 🚀")

if run:
    if not tweet.strip():
        st.warning("Please type a tweet first.")
    else:
        cleaned = clean_tweet(tweet)
        st.caption(f"Cleaned text passed to models: `{cleaned}`")

        need_svm = model_choice in ("Both (compare)", "SVC only")
        need_rnn = model_choice in ("Both (compare)", "SimpleRNN only")

        cols = st.columns(int(need_svm) + int(need_rnn))
        col_i = 0

        if need_svm:
            try:
                pipeline, le = load_svm()
                label, probs, classes = svm_predict(pipeline, le, cleaned)
                with cols[col_i]:
                    render_result("SVC (TF-IDF + LinearSVC)", label, probs, classes)
                col_i += 1
            except FileNotFoundError:
                st.error("Missing sentiment_svm_pipeline.pkl / label_encoder.pkl in the app folder.")

        if need_rnn:
            try:
                model = load_rnn()
                label, probs, classes = rnn_predict(model, cleaned)
                with cols[col_i]:
                    render_result("Bidirectional SimpleRNN", label, probs, classes)
            except (FileNotFoundError, OSError):
                st.error("Missing sentiment_rnn_model.keras in the app folder.")

st.markdown("---")
st.caption("SVC confidence is derived from decision-function scores (softmax'd), not true probabilities. RNN confidence is the model's softmax output.")