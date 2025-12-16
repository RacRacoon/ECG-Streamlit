import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

def lowpass(x, fc, fs, order=4):
    """Applies a zero-phase lowpass Butterworth filter."""
    if fc >= fs / 2: return x
    b, a = butter(order, fc/(fs/2), btype="low")
    return filtfilt(b, a, x)

def bandpass(x, f1, f2, fs, order=4):
    """Applies a zero-phase bandpass Butterworth filter."""
    if f1 >= fs / 2 or f2 >= fs / 2 or f1 >= f2: return x
    b, a = butter(order, [f1/(fs/2), f2/(fs/2)], btype="band")
    return filtfilt(b, a, x)

def moving_average(x, N):
    """Applies a moving average filter."""
    return np.convolve(x, np.ones(N)/N, mode="same")

def compute_dft(x, fs):
    """Computes the single-sided Discrete Fourier Transform (DFT)."""
    x = x - np.mean(x)
    N = len(x)
    X = np.abs(fft(x))[:N//2] * 2 / N
    f = fftfreq(N, 1/fs)[:N//2]
    return f, X

st.set_page_config(layout="wide")
st.title("ECG Processing Pipeline (Based on Diagram)")

# SIDEBAR CONTROLS
st.sidebar.header("File")
uploaded_file = st.sidebar.file_uploader("Upload ECG CSV (Time (s), ECG (V))", type=["csv"])

st.sidebar.header("Zoom")
t_start = st.sidebar.slider("Start Time (s)", 0.0, 30.0, 0.0)
window = st.sidebar.slider("Window Length (s)", 1.0, 10.0, 5.0)
amp_zoom = st.sidebar.slider("Amplitude Zoom", 0.5, 5.0, 1.0)

st.sidebar.header("Pre-filter")
use_lpf = st.sidebar.checkbox("Lowpass Filter", True)
fc_lp = st.sidebar.slider("LPF Cutoff (Hz)", 10.0, 50.0, 40.0)

st.sidebar.header("Bandpass Filter")
use_bpf = st.sidebar.checkbox("Bandpass Filter", True)
f_low = st.sidebar.slider("BPF Low Cut (Hz)", 0.5, 10.0, 5.0)
f_high = st.sidebar.slider("BPF High Cut (Hz)", 10.0, 30.0, 15.0)
bpf_stage = st.sidebar.slider("BPF Stages", 1, 5, 2)

df = None

if uploaded_file is None:
    st.info("Please upload an ECG CSV file to start the analysis.")
    st.stop() # Stops execution gracefully in Streamlit environment
    
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()
    
if df is None:
    st.error("Data loading failed and df is not defined.")
    st.stop()


# Validate and extract necessary columns
if "Time (s)" not in df.columns or "ECG (V)" not in df.columns:
    st.error("CSV must contain 'Time (s)' and 'ECG (V)' columns.")
    st.stop()

time = df["Time (s)"].values
ecg_raw = df["ECG (V)"].values
# Calculate sampling frequency
fs = 1 / (time[1] - time[0])


# ZOOM
t_end = min(t_start + window, time[-1])
idx = np.where((time >= t_start) & (time <= t_end))
t = time[idx]
ecg = ecg_raw[idx]

if len(t) == 0:
    st.warning("Zoom window is outside the data range. Please adjust Start Time or Window Length.")
    st.stop()


# PRE-FILTER + BANDPASS PIPELINE

ecg_f = ecg.copy()

if use_lpf:
    ecg_f = lowpass(ecg_f, fc_lp, fs)

if use_bpf:
    for _ in range(bpf_stage):
        ecg_f = bandpass(ecg_f, f_low, f_high, fs)


# PAN–TOMPKINS CORE

ecg_sq = ecg_f ** 2
mav = moving_average(ecg_sq, int(0.15 * fs))

threshold = 0.5 * np.max(mav)
peak_distance_samples = int(0.4 * fs)
peaks, _ = find_peaks(mav, height=threshold, distance=peak_distance_samples)

# RR & HR calculation
rr = np.diff(peaks) / fs
hr = 60 / np.mean(rr) if len(rr) > 0 else 0


# PLOTS

c1, c2 = st.columns(2)

# --- Plotting function helper ---
def plot_signal(t, y, title, amp_zoom, y_label="ECG (V)"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, y)
    max_abs_y = np.max(np.abs(y))
    if max_abs_y > 1e-6:
        ax.set_ylim(-amp_zoom * max_abs_y, amp_zoom * max_abs_y)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.grid(True)
    return fig

with c1:
    st.subheader("Raw ECG")
    fig = plot_signal(t, ecg, "Raw ECG", amp_zoom)
    st.pyplot(fig)

with c2:
    st.subheader("Filtered ECG")
    fig = plot_signal(t, ecg_f, "Filtered ECG", amp_zoom)
    st.pyplot(fig)

with c1:
    st.subheader("DFT of Raw ECG")
    f, X = compute_dft(ecg, fs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f[f < 40], X[f < 40]) # Plot only up to 40 Hz
    ax.set_title("DFT of Raw ECG")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig)

with c2:
    st.subheader("MAV + R-Peak Detection")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, mav, label="Moving Average of Squared Filtered ECG")
    ax.plot(t[peaks], mav[peaks], "ro", marker="|", markersize=20, label="Detected R-Peaks")
    ax.axhline(threshold, linestyle="--", color="gray", label=f"Detection Threshold ({threshold:.3f})")
    ax.set_title("MAV + R-Peak Detection")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude Squared")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.success(f"Estimated Heart Rate: **{hr:.2f} BPM**")