# %%
from lib_dynatree import DynatreeMeasurement
from lib_damping import DynatreeDampedSignal
import matplotlib.pyplot as plt
import numpy as np
import lib_dynatree
import logging
import pywt

lib_dynatree.logger.setLevel(logging.INFO)

m = DynatreeMeasurement(day="2021-03-22", tree="BK04", measurement="M02", measurement_type="normal", datapath="../data")
#
# m.data_acc5000.plot()
# plt.show()

# [2024-11-01 15:37:59,830] INFO | FFT finished in 0.19511151313781738
# [2024-11-01 15:37:59,830] INFO | peak is at frequency 0.23834114562644 Hz
# [2024-11-01 15:38:05,426] INFO | CWT finished in 5.7913031578063965
# [2024-11-01 15:38:18,071] INFO | wavelet normalize finished in 18.436726570129395, the norm is 88.69561339295582
# frequency = 0.23834114562644
# dt = 0.0002
# wavelet = "cmor1-1.5"
# # scale = pywt.frequency2scale(wavelet, [freq * dt])
# # coef, freqs = pywt.cwt(data, scale, wavelet,
# #                        sampling_period=dt)
#
# time = np.arange(0, 120, dt)
# data = np.cos(2 * np.pi * frequency * time)
# scale = pywt.frequency2scale(wavelet, [frequency * dt])
# scale = scale[0]
# # coef, freqs = pywt.cwt(data, scale, wavelet,
# #                        sampling_period=dt)
# # coef = np.abs(coef)[0, :]
# # coef.max()
#
#
# # Určíme celkovou dobu signálu a počet bodů
# duration = scale * dt
# num_points = int(duration / dt)
# t = np.linspace(-duration / 2, duration / 2, num_points)
#
# # Použijeme pywt k vytvoření waveletu typu "cmor"
# wavelet = pywt.ContinuousWavelet('cmor1-1.5')
#
# # Generujeme vlnku a normalizujeme čas podle dt
# psi,x = wavelet.wavefun(length=num_points)
# # wavelet_function = psi * np.exp(2j * np.pi * frequency * t / dt)
#
# # Vykreslíme výsledný wavelet
# plt.plot(x, np.abs(psi), label="Norma")
# plt.plot(x, np.real(psi), label="Reálná část")
# plt.plot(x, np.imag(psi), label="Imaginární část")
# plt.legend()
# plt.xlabel("Čas (s)")
# plt.ylabel("Amplitude")
# plt.title("Wavelet cmor1-1.5")
# plt.show()

# %%
s = DynatreeDampedSignal(measurement=m, signal_source="a02_z", #dt=0.0002,
                         # damped_start_time=54
                         )
plt.plot(s.damped_time, s.damped_signal)
plt.show()

# %%
s.damped_time
# %%

data,k,q = s.hilbert_envelope.values()
data,k,q

t = s.damped_time
ax = plt.plot(t, s.damped_signal)
plt.plot(t,data)
plt.plot(t, np.exp(k*t+q))
plt.show()


# %%

data,k,q, f1,f2 = s.wavelet_envelope.values()
# %%
t = s.damped_time
ax = plt.plot(t, s.damped_signal)
# plt.plot(t,data[:-1])
plt.plot(t, np.exp(k*t+q))
plt.show()

# %%
import plotly.graph_objects as go

def draw_signal_with_envelope(s, envelope=None, k=0, q=0, ):
    signal, time = s.damped_signal.reshape(-1), s.damped_time
    fig = go.Figure()
    x = time
    y = np.exp(k * time + q)
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([y, -y[::-1]]),
                             fill='toself',
                             fillcolor='lightblue',
                             line=dict(color='lightblue'),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='signal', line=dict(color='blue')))
    if envelope is not None:
        fig.add_trace(
            go.Scatter(x=time, y=envelope, mode='lines', name='envelope', line=dict(color='red'), legendgroup='obalka'))
        fig.add_trace(go.Scatter(x=time, y=-envelope, mode='lines', showlegend=False, line=dict(color='red'),
                                 legendgroup='obalka'))
    fig.update_layout(xaxis_title="Čas", yaxis_title="Signál")
    return fig

peaks,k,q = s.fit_maxima.values()
fig = draw_signal_with_envelope(s,k=k, q=q)
fig.add_trace(go.Scatter(x=peaks.index, y=peaks.values.reshape(-1),
                         mode='markers', name='peaks'))
fig.update_layout(title=f"Proložení exponenciely podle peaků, k={k:.4f}<br>{m}",
    # height = s.height.value,
    # width = s.width.value,
    )

# %%
peaks.plot(marker="o")

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=peaks.index, y=peaks.values,
                         mode='markers', name='peaks'))


# %%
peaks

# %%
peaks.values

# %%
