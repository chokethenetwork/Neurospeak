"""
Neuroprompting Streamlit Prototype
---------------------------------
Simulates EEG "thought" patterns and translates them into prompts for an LLM (e.g., GPT‑5).

Features
- Press a button to simulate an EEG sample from a small synthetic dataset of "thought intents".
- Visualize the simulated EEG (per‑channel time series + spectrogram‑ish summary).
- Classify the EEG into an intent using a simple nearest‑centroid model.
- Translate the intent into a concrete LLM prompt (text or image generation use cases).
- (Optional) Call OpenAI's API if OPENAI_API_KEY is set; otherwise, display the prompt you'd send.
- Allow uploading your own lightweight EEG feature CSV to replace the synthetic dataset.

Notes
- This is a quick prototype: not medical or BCI‑grade. Replace the classifier with your own model once you have real features.
- EEG here is simulated as 32 channels × 256 timesteps.

Run
$ pip install streamlit openai numpy pandas scipy altair pydub
$ streamlit run app.py
"""

from __future__ import annotations
import os
import io
import json
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.signal import welch

try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False

# ---------------------------
# Configuration & Constants
# ---------------------------
N_CHANNELS = 32
N_TIMESTEPS = 256
FS = 128  # Hz (simulated)
INTENTS = [
    "write_short_email",
    "summarize_article",
    "ask_research_question",
    "brainstorm_startup_ideas",
    "generate_image_animal",
    "generate_image_logo",
    "compose_tweet_thread",
    "translate_to_french",
]

INTENT_TEMPLATES: Dict[str, str] = {
    "write_short_email": (
        "You are a helpful assistant. Draft a short, clear email in under 120 words.\n"
        "Goal: {goal}\n"
        "Recipient: {recipient}\n"
        "Tone: {tone}\n"
        "Constraints: {constraints}\n"
    ),
    "summarize_article": (
        "Summarize the following article for a busy reader.\n"
        "Style: bullet points with 5 key takeaways + 1‑sentence TL;DR.\n"
        "Article: {context}\n"
    ),
    "ask_research_question": (
        "Formulate a precise research question and a brief plan (3 steps) to investigate it.\n"
        "Domain: {domain}\n"
        "Background: {context}\n"
    ),
    "brainstorm_startup_ideas": (
        "Brainstorm 5 startup ideas with target users, pain point, unique angle, and first MVP.\n"
        "Theme: {theme}\n"
    ),
    "generate_image_animal": (
        "Create a rich, concise text prompt for an image generator describing an animal scene.\n"
        "Animal: {animal}\n"
        "Style: {style}\n"
        "Scene constraints: {constraints}\n"
        "Return only the final prompt text.\n"
    ),
    "generate_image_logo": (
        "Write a tight image prompt for a minimalist startup logo.\n"
        "Company: {company}\n"
        "Vibe: {vibe}\n"
        "Colors: {colors}\n"
        "Return only the prompt.\n"
    ),
    "compose_tweet_thread": (
        "Draft a 6‑tweet thread (<=280 chars each) about {topic}, with hooks and plain language.\n"
    ),
    "translate_to_french": (
        "Translate the text to French, preserving meaning and simple tone.\n"
        "Text: {text}\n"
    ),
}

DEFAULT_FILLERS = {
    "goal": "Ask for a quick project update and suggest a short call tomorrow.",
    "recipient": "Project teammate",
    "tone": "friendly and efficient",
    "constraints": "avoid jargon",
    "context": "(paste article here)",
    "domain": "AI for agriculture",
    "theme": "offline AI tools for livestock farmers in Africa",
    "animal": "a dairy cow",
    "style": "photorealistic, early morning light",
    "company": "MlimisiAI",
    "vibe": "trustworthy, modern",
    "colors": "indigo, white",
    "topic": "EEG‑to‑prompt interfaces",
    "text": "Hello, how are you?",
}

# ---------------------------
# Synthetic EEG Dataset
# ---------------------------
@dataclass
class EEGSample:
    X: np.ndarray  # shape (channels, timesteps)
    y: str         # intent label


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def make_intent_centroids(n_channels=N_CHANNELS, n_timesteps=N_TIMESTEPS) -> Dict[str, np.ndarray]:
    """Create stable per‑intent centroids with different spectral fingerprints."""
    centroids = {}
    t = np.arange(n_timesteps) / FS
    for i, intent in enumerate(INTENTS):
        # Give each intent a dominant frequency band & phase shift
        base_freq = 6 + 2 * (i % 5)  # 6,8,10,12,14 Hz cycling
        phase = (i * np.pi / 8.0)
        centroid = []
        for ch in range(n_channels):
            freq = base_freq + 0.2 * ch
            signal = np.sin(2 * np.pi * freq * t + phase)
            signal += 0.3 * np.sin(2 * np.pi * (freq/2) * t)
            signal += 0.15 * np.sin(2 * np.pi * (freq*1.5) * t)
            centroid.append(signal)
        centroids[intent] = np.stack(centroid, axis=0)
    return centroids


def sample_from_centroid(centroid: np.ndarray, noise_scale: float = 0.5) -> np.ndarray:
    noise = noise_scale * np.random.randn(*centroid.shape)
    return centroid + noise


def make_synthetic_dataset(n_per_intent: int = 40, noise_scale: float = 0.5) -> List[EEGSample]:
    centroids = make_intent_centroids()
    data: List[EEGSample] = []
    for intent, C in centroids.items():
        for _ in range(n_per_intent):
            X = sample_from_centroid(C, noise_scale)
            data.append(EEGSample(X=X, y=intent))
    return data


# ---------------------------
# Lightweight Classifier
# ---------------------------
class NearestCentroid:
    def __init__(self):
        self.centroids: Dict[str, np.ndarray] = {}

    def fit(self, samples: List[EEGSample]):
        by_label: Dict[str, List[np.ndarray]] = {}
        for s in samples:
            by_label.setdefault(s.y, []).append(s.X)
        for label, arrs in by_label.items():
            self.centroids[label] = np.mean(np.stack(arrs, axis=0), axis=0)

    def predict_proba(self, X: np.ndarray) -> Dict[str, float]:
        # Use inverse distance as a similarity score, softmax to get probabilities.
        dists = {}
        for label, C in self.centroids.items():
            d = np.linalg.norm((X - C).ravel())
            dists[label] = d
        # Convert distances to similarities
        sims = {k: 1.0 / (v + 1e-6) for k, v in dists.items()}
        # Softmax
        vals = np.array(list(sims.values()))
        vals = np.exp(vals - vals.max())
        probs = vals / vals.sum()
        return {k: float(p) for k, p in zip(sims.keys(), probs)}


# ---------------------------
# Prompt Translation
# ---------------------------
@dataclass
class TranslationConfig:
    model_name: str = "gpt-5"
    temperature: float = 0.3
    max_tokens: int = 400


def template_prompt(intent: str, fillers: Dict[str, str]) -> str:
    tpl = INTENT_TEMPLATES.get(intent, "Explain the following intent: {intent}")
    try:
        return tpl.format(intent=intent, **{k: fillers.get(k, '') for k in DEFAULT_FILLERS})
    except Exception:
        # Fallback minimal prompt
        return f"User intent: {intent}. Produce a helpful response."


def call_openai_if_available(system: str, user: str, cfg: TranslationConfig) -> str:
    if not _openai_available:
        return "[OpenAI SDK not available in this environment. Showing the prompt instead.]\n\n" + user
    if not os.getenv("OPENAI_API_KEY"):
        return "[OPENAI_API_KEY not set. Showing the prompt instead.]\n\n" + user
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=cfg.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[OpenAI call failed: {e}]\n\n" + user


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Neuroprompting Prototype", page_icon="🧠", layout="wide")
seed_everything(42)

st.title("🧠 Neuroprompting Prototype: EEG → Prompt → LLM")
st.caption("Simulated EEG to intent classification, then prompt generation for text or images.")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Target", ["Text LLM", "Image Generator"], index=0)
    noise_scale = st.slider("EEG noise scale", 0.0, 1.5, 0.5, 0.05)
    n_per_intent = st.slider("Synthetic samples per intent (for training)", 5, 200, 40, 5)
    seed = st.number_input("Random seed", value=42, step=1)
    st.caption("Optional: upload your own EEG feature CSV (rows=samples, columns=features, plus 'label').")
    csv = st.file_uploader("Upload EEG CSV", type=["csv"]) 
    st.divider()
    st.header("LLM Config")
    model_name = st.text_input("Model name", "gpt-5")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05)
    max_tokens = st.slider("Max tokens", 64, 2000, 400, 16)

seed_everything(int(seed))

# Prepare data
if csv is not None:
    df = pd.read_csv(csv)
    if 'label' not in df.columns:
        st.error("CSV must include a 'label' column.")
        st.stop()
    labels = df['label'].unique().tolist()
    st.success(f"Loaded {len(df)} rows across {len(labels)} labels: {labels}")

    # Expect features flattened; reshape to (channels, timesteps) heuristically if possible
    feat_cols = [c for c in df.columns if c != 'label']
    feat_dim = len(feat_cols)
    if feat_dim != N_CHANNELS * N_TIMESTEPS:
        st.warning(
            f"Expected {N_CHANNELS*N_TIMESTEPS} features for reshape, got {feat_dim}. Will treat as flat vectors."
        )
        def to_sample(row):
            vec = row[feat_cols].values.astype(float)
            X = vec.reshape(1, -1)  # keep flat
            return EEGSample(X=X, y=row['label'])
    else:
        def to_sample(row):
            vec = row[feat_cols].values.astype(float)
            X = vec.reshape(N_CHANNELS, N_TIMESTEPS)
            return EEGSample(X=X, y=row['label'])

    samples = [to_sample(r) for _, r in df.iterrows()]
else:
    st.info("No CSV uploaded – using a synthetic EEG dataset.")
    samples = make_synthetic_dataset(n_per_intent=n_per_intent, noise_scale=noise_scale)

# Train classifier
clf = NearestCentroid()
clf.fit(samples)

colL, colR = st.columns([1.2, 1])
with colL:
    st.subheader("1) Simulate a 'thought' EEG sample")
    centroids = make_intent_centroids()
    picked_intent = st.selectbox("(Optional) Force ground‑truth intent for demo", ["random"] + INTENTS)
    if st.button("🧠 Think"):
        if picked_intent == "random":
            gt_intent = random.choice(INTENTS)
        else:
            gt_intent = picked_intent
        X = sample_from_centroid(centroids[gt_intent], noise_scale=noise_scale)
        st.session_state["last_eeg"] = X
        st.session_state["gt_intent"] = gt_intent

    X = st.session_state.get("last_eeg")
    gt_intent = st.session_state.get("gt_intent")

    if X is not None:
        # Time‑series overview for a few channels
        df_plot = pd.DataFrame({f"ch{c}": X[c] for c in range(min(N_CHANNELS, 8))})
        df_plot["t"] = np.arange(N_TIMESTEPS) / FS
        df_melt = df_plot.melt("t", var_name="channel", value_name="amp")
        chart = (
            alt.Chart(df_melt)
            .mark_line()
            .encode(x="t:Q", y="amp:Q", color="channel:N")
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)

        # Simple PSD summary (Welch)
        f, Pxx = welch(X, fs=FS, axis=1, nperseg=64)
        psd_df = pd.DataFrame({"f": f, "PSD": Pxx.mean(axis=0)})
        psd_chart = (
            alt.Chart(psd_df)
            .mark_line()
            .encode(x="f:Q", y="PSD:Q")
            .properties(height=180)
        )
        st.altair_chart(psd_chart, use_container_width=True)

        st.caption(f"Ground‑truth intent (hidden from classifier): {gt_intent}")
    else:
        st.info("Press **Think** to generate a sample.")

with colR:
    st.subheader("2) Classify intent & generate prompt")
    if X is None:
        st.stop()

    probs = clf.predict_proba(X)
    df_probs = pd.DataFrame(sorted(probs.items(), key=lambda x: -x[1]), columns=["intent", "probability"])    
    st.dataframe(df_probs, use_container_width=True, hide_index=True)

    top_intent = df_probs.iloc[0, 0]
    st.write(f"**Predicted intent:** `{top_intent}`")

    # Fillers UI
    with st.expander("Optional fields to enrich the prompt"):
        fillers = {}
        for k, v in DEFAULT_FILLERS.items():
            fillers[k] = st.text_input(k, v)

    # Translate to a concrete prompt
    user_prompt = template_prompt(top_intent, fillers)

    # For image vs text targets, we can wrap differently
    if mode == "Image Generator":
        system_msg = "You write precise, compact prompts for text‑to‑image models."
    else:
        system_msg = "You are a helpful AI assistant that turns user intents into actionable prompts."

    cfg = TranslationConfig(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

    with st.spinner("Translating intent → prompt via LLM (or showing the prompt)..."):
        llm_out = call_openai_if_available(system=system_msg, user=user_prompt, cfg=cfg)

    st.markdown("**LLM‑ready prompt:**")
    st.code(user_prompt)

    st.markdown("**LLM output (or echo if no API key):**")
    st.write(llm_out)

st.divider()
st.caption(
    "Privacy & Safety: This demo uses synthetic EEG and a toy classifier. Do not use for diagnosis or critical tasks."
)

os.environ["OPENAI_API_KEY"] = "ff6214b1f1d74a0aabe7942e17bf024e"
