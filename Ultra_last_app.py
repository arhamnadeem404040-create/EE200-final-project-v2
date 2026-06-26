import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
import os
import time
import pandas as pd
from collections import defaultdict
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------
# Page config — must be first
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Zapptain America || EE200",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme tokens ──
BG        = "#060c1a"
BG2       = "#0d1422"
BG3       = "#111827"
TEXT      = "#c9d1d9"
TEXT2     = "#6e7d8f"
ACCENT    = "#00d4ff"
ACCENT2   = "#0066ff"
BORDER    = "rgba(0,212,255,0.18)"
CARD_BG   = "rgba(13,20,35,0.9)"
GRAD_BG   = "radial-gradient(ellipse at 20% 50%, rgba(0,102,255,0.07) 0%,transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(0,212,255,0.05) 0%,transparent 50%)"
TAB_BG    = "rgba(13,20,35,0.8)"
SCORE_BAR = "linear-gradient(90deg,#1a3a5c,#0d2040)"
DIVIDER   = "rgba(0,212,255,0.12)"
FOOT_BG   = "rgba(0,212,255,0.04)"
FOOT_BOR  = "rgba(0,212,255,0.15)"
PLOT_BG   = "#060c1a"
PLOT_CARD = "#0d1422"
PLOT_TICK = "#6e7d8f"
PLOT_GRID = (1.0, 1.0, 1.0, 0.04)

# ---------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}
.stApp {{
    background: {BG};
    background-image: {GRAD_BG};
    min-height: 100vh;
}}
#MainMenu, footer, header {{visibility: hidden;}}
.block-container {{padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}}

/* ── Topbar ── */
.topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0 1rem;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 1.5rem;
}}
.topbar-left {{display:flex; flex-direction:column; gap:0.1rem;}}
.topbar-eyebrow {{
    font-family:'JetBrains Mono',monospace;
    font-size:0.62rem; letter-spacing:0.18em;
    color:{ACCENT}; text-transform:uppercase; opacity:0.8;
}}
.topbar-title {{
    font-family:'Space Grotesk',sans-serif;
    font-size:1.7rem; font-weight:700;
    background:linear-gradient(135deg,{ACCENT} 0%,{ACCENT2} 60%,#7c3aed 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    line-height:1.2;
}}
.topbar-sub {{font-size:0.78rem; color:{TEXT2}; letter-spacing:0.02em;}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background:{TAB_BG}; border:1px solid {BORDER};
    border-radius:10px; padding:4px; gap:4px; backdrop-filter:blur(10px);
}}
.stTabs [data-baseweb="tab"] {{
    font-family:'JetBrains Mono',monospace; font-size:0.72rem; font-weight:500;
    letter-spacing:0.12em; color:{TEXT2} !important; background:transparent !important;
    border-radius:7px !important; padding:0.5rem 1.4rem !important;
    border:none !important; transition:all 0.2s ease;
}}
.stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(0,102,255,0.15)) !important;
    color:{ACCENT} !important;
    border:1px solid {BORDER} !important;
    box-shadow:0 0 15px rgba(0,212,255,0.1);
}}
.stTabs [data-baseweb="tab-highlight"] {{display:none !important;}}
.stTabs [data-baseweb="tab-border"] {{display:none !important;}}

/* ── Cards ── */
.metric-card {{
    background:{CARD_BG}; border:1px solid {BORDER};
    border-radius:10px; padding:1rem 1.2rem; text-align:center;
    backdrop-filter:blur(10px); transition:border-color 0.2s, transform 0.15s;
}}
.metric-card:hover {{border-color:{ACCENT}; transform:translateY(-2px);}}
.metric-step {{
    font-family:'JetBrains Mono',monospace; font-size:0.6rem;
    color:{ACCENT}; letter-spacing:0.15em; margin-bottom:0.2rem; opacity:0.7;
}}
.metric-value {{
    font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700;
    background:linear-gradient(135deg,{ACCENT},{ACCENT2});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}}
.metric-label {{font-size:0.7rem; color:{TEXT2}; letter-spacing:0.05em; margin-top:0.1rem;}}

/* ── Stats dashboard cards ── */
.stat-dashboard {{
    display:grid; grid-template-columns:repeat(4,1fr); gap:0.8rem; margin:1rem 0;
}}
.stat-card {{
    background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px;
    padding:1rem 1rem 0.8rem; position:relative; overflow:hidden;
    transition:transform 0.2s, border-color 0.2s;
}}
.stat-card:hover {{transform:translateY(-3px); border-color:{ACCENT};}}
.stat-card::before {{
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,{ACCENT},{ACCENT2});
}}
.stat-icon {{font-size:1.4rem; margin-bottom:0.4rem; display:block;}}
.stat-val {{
    font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700;
    background:linear-gradient(135deg,{ACCENT},{ACCENT2},#7c3aed);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    line-height:1; margin-bottom:0.15rem;
}}
.stat-lbl {{font-size:0.72rem; color:{TEXT2}; text-transform:uppercase; letter-spacing:0.1em;}}
.stat-sub {{font-size:0.65rem; color:{ACCENT}; opacity:0.7; margin-top:0.1rem; font-family:'JetBrains Mono',monospace;}}

/* ── Mini bar chart row ── */
.bar-chart-row {{display:flex; align-items:flex-end; gap:4px; height:40px; margin:0.5rem 0;}}
.bar-seg {{
    flex:1; border-radius:3px 3px 0 0;
    background:linear-gradient(180deg,{ACCENT},{ACCENT2}); opacity:0.85;
    transition:opacity 0.2s;
}}
.bar-seg:hover {{opacity:1;}}

/* ── Match banner ── */
.match-banner {{
    background:linear-gradient(135deg,rgba(0,212,255,0.08) 0%,rgba(0,102,255,0.08) 100%);
    border:1px solid {BORDER}; border-left:4px solid {ACCENT};
    border-radius:10px; padding:1.5rem 2rem; margin:1.5rem 0;
}}
.match-label {{
    font-family:'JetBrains Mono',monospace; font-size:0.65rem;
    letter-spacing:0.2em; color:{ACCENT}; text-transform:uppercase; margin-bottom:0.4rem;
}}
.match-title {{
    font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700;
    color:{TEXT}; margin:0 0 0.4rem;
}}
.match-sub {{font-size:0.9rem; color:{TEXT2};}}
.match-sub span {{color:#f0a500; font-weight:600;}}

/* ── Score bars ── */
.score-row {{
    display:flex; align-items:center; gap:1rem; padding:0.5rem 0;
    border-bottom:1px solid {DIVIDER};
}}
.score-name {{font-size:0.9rem; color:{TEXT}; min-width:220px; flex-shrink:0;}}
.score-bar-wrap {{flex:1; background:rgba(128,128,128,0.1); border-radius:4px; height:6px; overflow:hidden;}}
.score-bar-fill {{height:100%; border-radius:4px; background:linear-gradient(90deg,{ACCENT},{ACCENT2});}}
.score-num {{font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:{ACCENT}; min-width:60px; text-align:right;}}

/* ── Step blocks ── */
.step-block {{
    border-left:3px solid rgba(0,212,255,0.4); padding:0.8rem 1.2rem;
    margin:1.5rem 0 0.8rem; background:rgba(0,212,255,0.03); border-radius:0 8px 8px 0;
}}
.step-eyebrow {{
    font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:{ACCENT};
    letter-spacing:0.18em; text-transform:uppercase; margin-bottom:0.15rem; opacity:0.8;
}}
.step-heading {{
    font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:600;
    color:{TEXT}; margin:0 0 0.3rem;
}}
.step-body {{font-size:0.88rem; color:{TEXT2}; line-height:1.6;}}
.step-body b {{color:{ACCENT}; font-weight:600;}}

/* ── Library cards ── */
.lib-song {{
    font-family:'Space Grotesk',sans-serif; font-size:0.82rem; font-weight:600;
    color:{TEXT}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}}
.lib-hashes {{font-family:'JetBrains Mono',monospace; font-size:0.68rem;}}

/* ── Section labels ── */
.section-eyebrow {{
    font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:0.2em;
    color:{ACCENT}; text-transform:uppercase; margin-bottom:0.3rem; opacity:0.8;
}}
.section-title {{
    font-family:'Space Grotesk',sans-serif; font-size:1.4rem; font-weight:600;
    color:{TEXT}; margin:0 0 0.6rem;
}}
.section-body {{font-size:0.9rem; color:{TEXT2}; line-height:1.6; margin-bottom:1.2rem;}}

/* ── Upload zone ── */
.stFileUploader > div > div {{
    background:{CARD_BG} !important; border:2px dashed {BORDER} !important;
    border-radius:10px !important; transition:border-color 0.2s;
}}
.stFileUploader > div > div:hover {{border-color:{ACCENT} !important;}}

/* ── Buttons ── */
.stButton > button {{
    background:linear-gradient(135deg,rgba(0,212,255,0.12),rgba(0,102,255,0.12)) !important;
    border:1px solid {BORDER} !important; color:{ACCENT} !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
    font-weight:500 !important; letter-spacing:0.1em !important;
    border-radius:8px !important; padding:0.45rem 1.2rem !important;
    transition:all 0.2s !important; text-transform:uppercase !important;
}}
.stButton > button:hover {{
    border-color:{ACCENT} !important;
    box-shadow:0 0 18px rgba(0,212,255,0.18) !important;
    transform:translateY(-1px) !important;
}}

/* ── How-it-works ── */
.how-card {{
    background:{CARD_BG}; border:1px solid {BORDER}; border-radius:12px;
    padding:1.4rem; text-align:center; height:100%;
}}
.how-icon {{font-size:2rem; margin-bottom:0.6rem;}}
.how-num {{font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:{ACCENT}; letter-spacing:0.15em; margin-bottom:0.3rem; opacity:0.7;}}
.how-title {{font-family:'Space Grotesk',sans-serif; font-size:0.95rem; font-weight:600; color:{TEXT}; margin-bottom:0.4rem;}}
.how-body {{font-size:0.8rem; color:{TEXT2}; line-height:1.5;}}

/* ── Info / no-match ── */
.info-box {{
    background:rgba(0,212,255,0.05); border:1px solid {BORDER};
    border-radius:8px; padding:1rem 1.2rem; font-size:0.88rem; color:{TEXT2}; line-height:1.6; margin-bottom:1rem;
}}
.info-box code {{background:rgba(0,212,255,0.1); color:{ACCENT}; padding:0.1rem 0.4rem; border-radius:4px; font-family:'JetBrains Mono',monospace; font-size:0.8rem;}}
.no-match {{
    background:rgba(200,50,50,0.08); border:1px solid rgba(200,50,50,0.3);
    border-left:4px solid #e05050; border-radius:10px;
    padding:1.2rem 1.5rem; margin:1rem 0; color:#e05050; font-size:1rem;
}}

/* ── Loader ── */
@keyframes spin {{ to {{transform:rotate(360deg);}} }}
@keyframes pulse-ring {{
    0%   {{transform:scale(0.8); opacity:0.8;}}
    50%  {{transform:scale(1.1); opacity:0.4;}}
    100% {{transform:scale(0.8); opacity:0.8;}}
}}
@keyframes fade-in {{ from{{opacity:0;transform:translateY(8px);}} to{{opacity:1;transform:translateY(0);}} }}
.loader-wrap {{
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; gap:1rem; padding:2.5rem 0;
    animation:fade-in 0.3s ease;
}}
.loader-ring {{
    width:56px; height:56px; border-radius:50%;
    border:3px solid {BG3}; border-top-color:{ACCENT};
    animation:spin 0.9s linear infinite; position:relative;
}}
.loader-ring::before {{
    content:''; position:absolute; inset:-8px; border-radius:50%;
    border:2px solid {ACCENT}; opacity:0.2; animation:pulse-ring 1.5s ease-in-out infinite;
}}
.loader-text {{
    font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    color:{ACCENT}; letter-spacing:0.15em; text-transform:uppercase;
}}
.loader-steps {{display:flex; gap:0.5rem;}}
.loader-dot {{
    width:6px; height:6px; border-radius:50%; background:{ACCENT}; opacity:0.3;
    animation:pulse-ring 1.2s ease-in-out infinite;
}}
.loader-dot:nth-child(2) {{animation-delay:0.2s;}}
.loader-dot:nth-child(3) {{animation-delay:0.4s;}}

/* ── Divider ── */
.cyan-divider {{border:none; border-top:1px solid {DIVIDER}; margin:1.5rem 0;}}

/* ── Footer ── */
.footer {{
    margin-top:3rem; padding:1.5rem 2rem;
    background:{FOOT_BG}; border-top:1px solid {FOOT_BOR};
    border-radius:10px; text-align:center;
}}
.footer-made {{
    font-family:'JetBrains Mono',monospace; font-size:0.72rem;
    color:{TEXT2}; letter-spacing:0.08em; margin-bottom:0.4rem;
}}
.footer-names {{
    font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:600;
    background:linear-gradient(135deg,{ACCENT} 0%,{ACCENT2} 50%,#7c3aed 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}}
.footer-sub {{font-size:0.72rem; color:{TEXT2}; margin-top:0.3rem; opacity:0.7;}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Core fingerprinting logic
# ---------------------------------------------------------------
def get_fingerprints_with_time(audio_input, sr=None, is_path=True):
    if is_path:
        audio, sr = librosa.load(audio_input, sr=None)
    else:
        audio = audio_input
    D = librosa.stft(audio, n_fft=2048, hop_length=512)
    S = np.abs(D)
    local_max = maximum_filter(S, size=20)
    threshold = np.percentile(S, 99.5)
    peaks = (S == local_max) & (S > threshold)
    coords = np.argwhere(peaks)
    coords = coords[coords[:, 0] > 10]
    coords = coords[np.argsort(coords[:, 1])]
    hashes = []
    for i in range(len(coords)):
        f1, t1 = coords[i]
        for j in range(i + 1, min(i + 15, len(coords))):
            f2, t2 = coords[j]
            h = (int(f1), int(f2), int(t2 - t1))
            hashes.append((h, int(t1)))
    return hashes, D, coords


def match_query(query_hashes_with_time, hash_db):
    offset_counts = defaultdict(lambda: defaultdict(int))
    for h, t_query in query_hashes_with_time:
        if h in hash_db:
            for song, t_song in hash_db[h]:
                offset = t_song - t_query
                offset_counts[song][offset] += 1
    scores = {}
    for song, hist in offset_counts.items():
        best_offset = max(hist, key=hist.get)
        scores[song] = hist[best_offset]
    return scores, offset_counts


@st.cache_resource
def load_database():
    with open("hash_db.pkl", "rb") as f:
        return pickle.load(f)


def identify_clip(audio, sr, hash_db):
    t0 = time.time()
    hashes, D, coords = get_fingerprints_with_time(audio, sr=sr, is_path=False)
    t_features = (time.time() - t0) * 1000
    t1 = time.time()
    scores, offset_counts = match_query(hashes, hash_db)
    t_match = (time.time() - t1) * 1000
    if not scores:
        return None, None, D, coords, {}, 0, 0, len(hashes)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    matched_song = sorted_scores[0][0]
    best_hist = offset_counts[matched_song]
    return matched_song, sorted_scores, D, coords, best_hist, t_features, t_match, len(hashes)


# ---------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------
def dark_fig(w=8, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PLOT_CARD)
    ax.set_facecolor(PLOT_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_TICK); spine.set_linewidth(0.6)
    ax.tick_params(colors=PLOT_TICK, labelsize=8)
    ax.xaxis.label.set_color(PLOT_TICK)
    ax.yaxis.label.set_color(PLOT_TICK)
    ax.title.set_color(TEXT)
    ax.grid(color=PLOT_GRID, linewidth=0.5)
    return fig, ax

def plot_spectrogram(D, sr):
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = dark_fig(w=8, h=3.5)
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.set_title("Spectrogram · time-frequency map", fontsize=9, pad=8)
    fig.tight_layout(); return fig

def plot_constellation(coords, D, sr):
    fig, ax = dark_fig(w=8, h=3.5)
    times = coords[:, 1] * 512 / sr
    freqs = coords[:, 0] * sr / D.shape[0] / 2
    ax.scatter(times, freqs, s=6, color=ACCENT, alpha=0.7, linewidths=0)
    ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel("Frequency (Hz)", fontsize=8)
    ax.set_title(f"Constellation map · {len(coords)} peaks", fontsize=9, pad=8)
    fig.tight_layout(); return fig

def plot_offset_histogram(hist, song_name):
    fig, ax = dark_fig(w=8, h=3.2)
    if hist:
        offsets = list(hist.keys()); counts = list(hist.values())
        peak_offset = max(hist, key=hist.get)
        colors = [ACCENT if o == peak_offset else "#1a3a5c" for o in offsets]
        ax.bar(offsets, counts, width=1, color=colors, linewidth=0)
        ax.axvline(peak_offset, color="#f0a500", linewidth=1.5, linestyle="--", alpha=0.8)
    ax.set_xlabel("Offset (time bins)", fontsize=8); ax.set_ylabel("Matching hashes", fontsize=8)
    ax.set_title("Offset histogram · alignment spike", fontsize=9, pad=8)
    fig.tight_layout(); return fig


# ---------------------------------------------------------------
# TOPBAR
# ---------------------------------------------------------------
st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-eyebrow">◆ EE200 || Signals, Systems &amp; Networks || Project </div>
    <div class="topbar-title">🎵 Zapptain America</div>
    <div class="topbar-sub">Audio Fingerprinting &amp; Song Recognition System</div>
  </div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Load DB
# ---------------------------------------------------------------
try:
    hash_db = load_database()
except FileNotFoundError:
    st.markdown('<div class="no-match">⚠ hash_db.pkl not found. Run your Q3A notebook to generate it.</div>', unsafe_allow_html=True)
    st.stop()

# Precompute library stats
song_hash_counts = defaultdict(int)
for hash_key, entries in hash_db.items():
    for song, t in entries:
        song_hash_counts[song] += 1
songs = sorted(song_hash_counts.keys())
total_hashes = sum(song_hash_counts.values())
avg_hashes = total_hashes // max(len(songs), 1)

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab_lib, tab_id, tab_batch = st.tabs(["🎶  LIBRARY", "◎  IDENTIFY", "≡  BATCH"])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — LIBRARY
# ═══════════════════════════════════════════════════════════════
with tab_lib:
    st.markdown(f"""
    <div class="section-eyebrow">DATABASE</div>
    <div class="section-title">Song Library</div>
    <div class="section-body">Every track has been converted into a constellation of spectrogram peaks and stored as compact hashes. Each thumbnail below is the actual fingerprint — a unique visual signature of that song.</div>
    """, unsafe_allow_html=True)

    max_hashes = max(song_hash_counts.values()) if song_hash_counts else 1
    top_song = max(song_hash_counts, key=song_hash_counts.get) if song_hash_counts else "—"
    top_clean = os.path.splitext(top_song)[0]

    st.markdown(f"""
    <div class="stat-dashboard">
      <div class="stat-card">
        <span class="stat-icon">🔊</span>
        <div class="stat-val">{len(songs)}</div>
        <div class="stat-lbl">Indexed Tracks</div>
        <div class="stat-sub">in database</div>
      </div>
      <div class="stat-card">
        <span class="stat-icon">🔑</span>
        <div class="stat-val">{total_hashes:,}</div>
        <div class="stat-lbl">Total Hashes</div>
        <div class="stat-sub">fingerprint anchors</div>
      </div>
      <div class="stat-card">
        <span class="stat-icon">📊</span>
        <div class="stat-val">{avg_hashes:,}</div>
        <div class="stat-lbl">Average per Track</div>
        <div class="stat-sub">hashes / song</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if len(songs) > 1:
        st.markdown("<hr class='cyan-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="section-eyebrow" style="margin-bottom:0.4rem;">HASH DISTRIBUTION</div>', unsafe_allow_html=True)
        bar_heights = [int(song_hash_counts[s] / max_hashes * 100) for s in songs]
        bars_html = "".join(
            f'<div class="bar-seg" style="height:{h}%;opacity:{0.5+h/200:.2f};" title="{os.path.splitext(songs[i])[0]}: {song_hash_counts[songs[i]]:,}"></div>'
            for i, h in enumerate(bar_heights)
        )
        st.markdown(f'<div class="bar-chart-row">{bars_html}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.68rem;color:{TEXT2};font-family:JetBrains Mono,monospace;">Each bar = one song · hover for name · height = relative hash count</div>', unsafe_allow_html=True)

    st.markdown("<hr class='cyan-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow" style="margin-bottom:0.8rem;">IN THE DATABASE</div>', unsafe_allow_html=True)

    palette = ["#00d4ff","#0088ff","#7c3aed","#00e5a0","#f0a500","#e05050","#ff6eb4","#44d9e8"]
    cols_per_row = 4
    for row_start in range(0, len(songs), cols_per_row):
        cols = st.columns(cols_per_row)
        for ci, song in enumerate(songs[row_start:row_start+cols_per_row]):
            color = palette[(row_start//cols_per_row + ci) % len(palette)]
            with cols[ci]:
                times_r, freqs_r = [], []
                for (f1, f2, dt), entries in hash_db.items():
                    for s, t in entries:
                        if s == song:
                            times_r.append(t); freqs_r.append(f1)
                            if len(times_r) > 800: break
                    if len(times_r) > 800: break
                fig2, ax2 = plt.subplots(figsize=(3, 1.8))
                fig2.patch.set_facecolor("#0a1020")
                ax2.set_facecolor("#0a1020")
                for sp in ax2.spines.values(): sp.set_visible(False)
                ax2.set_xticks([]); ax2.set_yticks([])
                if times_r:
                    ax2.scatter(times_r, freqs_r, s=1.2, color=color, alpha=0.55, linewidths=0)
                fig2.tight_layout(pad=0.1)
                st.pyplot(fig2, width='stretch')
                plt.close(fig2)
                clean_name = os.path.splitext(song)[0]
                h_count = song_hash_counts[song]
                donut_pct = int(h_count / max_hashes * 100)
                st.markdown(f"""
                <div style="margin-top:0.3rem;">
                  <div class="lib-song" title="{clean_name}">{clean_name}</div>
                  <div class="lib-hashes" style="color:{color};">{h_count:,} hashes &nbsp;·&nbsp; {donut_pct}% of max</div>
                </div>""", unsafe_allow_html=True)
                # ── Audio player ──
                song_path = song  # tries current directory first
                if os.path.exists(song_path):
                    with open(song_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — IDENTIFY
# ═══════════════════════════════════════════════════════════════
with tab_id:
    st.markdown(f"""
    <div class="section-eyebrow">SEARCH</div>
    <div class="section-title">Identify a clip</div>
    <div class="section-body">Upload a short audio snippet — even noisy or modified — and the system will fingerprint it and match it against the indexed library in real time.</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Query Audio File (.wav or .mp3)",
                                type=["mp3","wav"], label_visibility="collapsed")
    st.markdown(f'<div style="font-size:0.75rem;color:{TEXT2};margin-top:0.3rem;margin-bottom:1rem;">↑ WAV · MP3 · M4A &nbsp;·&nbsp; Any duration</div>', unsafe_allow_html=True)

    if uploaded is not None:
        loader = st.empty()
        loader.markdown("""
        <div class="loader-wrap">
          <div class="loader-ring"></div>
          <div class="loader-text">Fingerprinting audio…</div>
          <div class="loader-steps">
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
            <div class="loader-dot"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        audio, sr = librosa.load(uploaded, sr=None)
        matched_song, sorted_scores, D, coords, best_hist, t_feat, t_match, n_hashes = \
            identify_clip(audio, sr, hash_db)
        loader.empty()

        n_peaks = len(coords)
        total_ms = t_feat + t_match

        st.markdown("<hr class='cyan-divider'>", unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.7rem;color:{TEXT2};font-family:JetBrains Mono,monospace;text-align:right;margin-bottom:0.5rem;">total {total_ms:.0f} ms</div>', unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        timing_data = [
            ("① SPECTROGRAM", f"{t_feat*0.25:.0f} ms", f"{D.shape[0]}×{D.shape[1]}"),
            ("② CONSTELLATION", f"{t_feat*0.35:.0f} ms", f"{n_peaks} peaks"),
            ("③ HASHING", f"{t_feat*0.15:.0f} ms", f"{n_hashes:,} hashes"),
            ("④ DB LOOKUP", f"{t_match*0.7:.0f} ms", f"{len(sorted_scores or [])} tracks"),
            ("⑤ SCORING", f"{t_match*0.3:.0f} ms", "offset vote"),
        ]
        for col,(step,val,sub) in zip([m1,m2,m3,m4,m5], timing_data):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-step">{step}</div><div class="metric-value" style="font-size:1.1rem;">{val}</div><div class="metric-label">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("<hr class='cyan-divider'>", unsafe_allow_html=True)

        if matched_song is None:
            st.markdown('<div class="no-match">✕ No match found. Try a longer clip or check your hash_db.</div>', unsafe_allow_html=True)
        else:
            top_score = sorted_scores[0][1]
            runner_up = sorted_scores[1][1] if len(sorted_scores) > 1 else 1
            ratio = top_score / max(runner_up, 1)
            clean_match = os.path.splitext(matched_song)[0]

            st.markdown(f"""
            <div class="match-banner">
              <div class="match-label">✓ Match Found</div>
              <div class="match-title">{clean_match}</div>
              <div class="match-sub">cluster score <span>{top_score:,}</span> &nbsp;·&nbsp; <span>{ratio:.0f}×</span> the runner-up</div>
            </div>""", unsafe_allow_html=True)

            qa, qb, qc, qd = st.columns(4)
            with qa:
                st.markdown(f'<div class="metric-card"><div class="metric-step">MATCH SCORE</div><div class="metric-value">{top_score:,}</div><div class="metric-label">hash votes</div></div>', unsafe_allow_html=True)
            with qb:
                st.markdown(f'<div class="metric-card"><div class="metric-step">CONFIDENCE</div><div class="metric-value">{ratio:.0f}×</div><div class="metric-label">vs runner-up</div></div>', unsafe_allow_html=True)
            with qc:
                st.markdown(f'<div class="metric-card"><div class="metric-step">PEAKS FOUND</div><div class="metric-value">{n_peaks}</div><div class="metric-label">constellation</div></div>', unsafe_allow_html=True)
            with qd:
                st.markdown(f'<div class="metric-card"><div class="metric-step">HASHES USED</div><div class="metric-value">{n_hashes:,}</div><div class="metric-label">fingerprint pairs</div></div>', unsafe_allow_html=True)

            st.markdown("<hr class='cyan-divider'>", unsafe_allow_html=True)

            st.markdown(f'<div class="section-eyebrow" style="margin-bottom:0.6rem;">CANDIDATE SCORES</div>', unsafe_allow_html=True)
            max_s = sorted_scores[0][1]
            for song, score in sorted_scores[:7]:
                pct = score / max_s * 100
                clean = os.path.splitext(song)[0]
                is_top = song == matched_song
                bar_col = f"linear-gradient(90deg,{ACCENT},{ACCENT2})" if is_top else SCORE_BAR
                name_html = f"<b style='color:{TEXT}'>{clean}</b>" if is_top else clean
                num_html = f"<span style='color:{ACCENT}'>{score}</span>" if is_top else str(score)
                st.markdown(f"""
                <div class="score-row">
                  <div class="score-name">{name_html}</div>
                  <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{pct}%;background:{bar_col};"></div></div>
                  <div class="score-num">{num_html}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<hr class='cyan-divider'>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="step-block">
              <div class="step-eyebrow">Step 1 · Feature Extraction</div>
              <div class="step-heading">From spectrogram to constellation</div>
              <div class="step-body">The clip was converted into a time-frequency map (left) — brighter means louder at that frequency and moment. From that rich image, only the <b>{n_peaks} most prominent peaks</b> were kept (right). Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and noise.</div>
            </div>""", unsafe_allow_html=True)
            col_spec, col_const = st.columns(2)
            with col_spec:
                fig = plot_spectrogram(D, sr); st.pyplot(fig, width='stretch'); plt.close(fig)
            with col_const:
                fig = plot_constellation(coords, D, sr); st.pyplot(fig, width='stretch'); plt.close(fig)

            st.markdown(f"""
            <div class="step-block">
              <div class="step-eyebrow">Step 2 · Database Search &amp; Step 3 · The Proof</div>
              <div class="step-heading">The alignment spike</div>
              <div class="step-body">The <b>{n_hashes:,} fingerprint hashes</b> were looked up against every indexed track. Every matched hash votes for a time offset (database frame − query frame). Chance matches scatter votes randomly — a flat noise floor. A genuine match makes them converge: <b>{top_score:,} hashes agreed on a single offset</b>. That spike cannot be a coincidence.</div>
            </div>""", unsafe_allow_html=True)
            fig = plot_offset_histogram(best_hist, clean_match)
            st.pyplot(fig, width='stretch'); plt.close(fig)

    else:
        st.markdown("<hr class='cyan-divider'>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-eyebrow" style="margin-bottom:1rem;">HOW IT WORKS</div>', unsafe_allow_html=True)
        h1,h2,h3 = st.columns(3)
        how_items = [
            ("🎵","01 · AUDIO DATABASE","Reference songs","The library of MP3 tracks is fingerprinted once and indexed into a hash database."),
            ("⬆️","02 · UPLOAD QUERY","Drop a clip","Upload any short, noisy, or modified audio snippet you want to recognise."),
            ("🔍","03 · IDENTIFY SONG","Instant match","Click Identify to match the query against the database using offset histogram voting."),
        ]
        for col,(icon,num,title,body) in zip([h1,h2,h3], how_items):
            with col:
                st.markdown(f'<div class="how-card"><div class="how-icon">{icon}</div><div class="how-num">{num}</div><div class="how-title">{title}</div><div class="how-body">{body}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — BATCH
# ═══════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown(f"""
    <div class="section-eyebrow">BATCH</div>
    <div class="section-title">Identify many clips at once</div>
    <div class="section-body">Upload a set of query clips. Each is identified against the currently indexed library, and the results are written to a standardised <code>results.csv</code> with columns <code>filename</code>, <code>prediction</code>. The prediction is the matched track's filename without its extension, or <code>none</code> when no candidate clears the confidence threshold.</div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload query clips", type=["mp3","wav"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown(f'<div style="font-size:0.75rem;color:{TEXT2};margin-top:0.3rem;margin-bottom:1rem;">↑ WAV · MP3 · M4A &nbsp;·&nbsp; Multiple files</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.markdown(f'<div class="info-box">{len(uploaded_files)} file{"s" if len(uploaded_files)>1 else ""} ready · click <b>Run Batch</b> to identify all</div>', unsafe_allow_html=True)

    if uploaded_files and st.button("▶  Run Batch"):
        results = []
        prog = st.progress(0, text="Identifying clips…")
        status_box = st.empty()
        for i, f in enumerate(uploaded_files):
            status_box.markdown(f'<div class="loader-wrap"><div class="loader-ring"></div><div class="loader-text">Processing {f.name} &nbsp;({i+1}/{len(uploaded_files)})</div></div>', unsafe_allow_html=True)
            audio, sr = librosa.load(f, sr=None)
            matched_song, *_ = identify_clip(audio, sr, hash_db)
            prediction = os.path.splitext(matched_song)[0] if matched_song else "none"
            results.append({"filename": f.name, "prediction": prediction})
            prog.progress((i+1)/len(uploaded_files), text=f"Identified {i+1}/{len(uploaded_files)}")
        status_box.empty(); prog.empty()

        results_df = pd.DataFrame(results)
        matched = sum(1 for r in results if r["prediction"] != "none")
        pct = int(matched/len(results)*100) if results else 0

        st.markdown("<hr class='cyan-divider'>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-step">TOTAL CLIPS</div><div class="metric-value">{len(results)}</div><div class="metric-label">processed</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-step">MATCHED</div><div class="metric-value" style="background:linear-gradient(135deg,#00d4ff,#00e5a0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{matched}</div><div class="metric-label">songs identified</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-step">UNMATCHED</div><div class="metric-value" style="background:linear-gradient(135deg,#e05050,#f0a500);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{len(results)-matched}</div><div class="metric-label">no match</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-step">MATCH RATE</div><div class="metric-value">{pct}%</div><div class="metric-label">accuracy</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(results_df, width='stretch', hide_index=True)
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇  Download results.csv", csv_bytes, file_name="results.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("<hr class='cyan-divider' style='margin-top:3rem;'>", unsafe_allow_html=True)
st.markdown(f"""
<div class="footer">
  <div class="footer-made">MADE WITH ❤️ BY</div>
  <div class="footer-names">Arham &nbsp;·&nbsp; Roll: 250669 &nbsp;&nbsp;|&nbsp;&nbsp; Ayan &nbsp;·&nbsp; Roll: 250670</div>
  <div class="footer-sub"> EE200 | Signals, Systems &amp; Networks | IIT Kanpur </div>
</div>
""", unsafe_allow_html=True)
