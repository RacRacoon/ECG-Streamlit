import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq

# parameters fro Streamlit
FILE_NAME = "exampleECG.csv"
t_start = 0.0
window = 5.0
amp_zoom = 1.0
use_lpf = True
fc_lp = 40.0
use_bpf = True
f_low = 5.0
f_high = 15.0
bpf_stage = 2

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

# =====================================================
# LOAD DATA
# =====================================================
try:
    df = pd.read_csv(FILE_NAME)
    if "Time (s)" not in df.columns or "ECG (V)" not in df.columns:
        raise ValueError("CSV must contain 'Time (s)' and 'ECG (V)' columns.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

time = df["Time (s)"].values
ecg_raw = df["ECG (V)"].values
fs = 1 / (time[1] - time[0])

t_end = min(t_start + window, time[-1])
idx = np.where((time >= t_start) & (time <= t_end))
t = time[idx]
ecg = ecg_raw[idx]

if len(t) == 0:
    print("Error: Zoom window is outside the data range.")
    exit()


ecg_f = ecg.copy()

if use_lpf:
    ecg_f = lowpass(ecg_f, fc_lp, fs)

if use_bpf:
    for _ in range(bpf_stage):
        ecg_f = bandpass(ecg_f, f_low, f_high, fs)


ecg_sq = ecg_f ** 2
mav = moving_average(ecg_sq, int(0.15 * fs))

threshold = 0.5 * np.max(mav)
peak_distance_samples = int(0.4 * fs)
peaks, _ = find_peaks(mav, height=threshold, distance=peak_distance_samples)

# RR & HR calculation
rr = np.diff(peaks) / fs
hr = 60 / np.mean(rr) if len(rr) > 0 else 0

# Plotting
def create_plot(t, y, title, amp_zoom, y_label="ECG (V)"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, y)
    max_abs_y = np.max(np.abs(y))
    if max_abs_y > 1e-6:
        ax.set_ylim(-amp_zoom * max_abs_y, amp_zoom * max_abs_y)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.tight_layout()
    return fig

# Raw ECG
fig1 = create_plot(t, ecg, "Raw ECG (Zoomed)", amp_zoom)
fig1.savefig("raw_ecg_zoomed.png")
plt.close(fig1)

# Filtered ECG
fig2 = create_plot(t, ecg_f, "Filtered ECG (Zoomed)", amp_zoom)
fig2.savefig("filtered_ecg_zoomed.png")
plt.close(fig2)

# DFT of Raw ECG
f, X = compute_dft(ecg, fs)
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(f[f < 40], X[f < 40])
ax3.set_title("DFT of Raw ECG")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Amplitude")
ax3.grid(True)
fig3.tight_layout()
fig3.savefig("dft_ecg.png")
plt.close(fig3)

# MAV + R-Peak Detection
fig4, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(t, mav, label="Moving Average of Squared Filtered ECG")
ax4.plot(t[peaks], mav[peaks], "ro", marker="|", markersize=20, label="Detected R-Peaks")
ax4.axhline(threshold, linestyle="--", color="gray", label=f"Detection Threshold ({threshold:.3f})")
ax4.set_title("MAV + R-Peak Detection")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Amplitude Squared")
ax4.legend()
ax4.grid(True)
fig4.tight_layout()
fig4.savefig("mav_r_peak_detection.png")
plt.close(fig4)

print(f"Estimated Heart Rate: {hr:.2f} BPM")