import numpy as np, wave, os, random
from scipy.signal import lfilter, butter
import deepavali as D

SR = 44100; N = int(SR * D.DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)

def put(sig, t, gain=1.0, pan=0.0):
    i = int(t * SR); j = min(N, i + len(sig))
    if i >= N or j <= i: return
    s = sig[:j - i] * gain
    L[i:j] += s * np.sqrt((1 - pan) / 2); R[i:j] += s * np.sqrt((1 + pan) / 2)

def tt(d): return np.arange(int(d * SR)) / SR
def bp(x, lo, hi): b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], "band"); return lfilter(b, a, x)
def lp(x, f): b, a = butter(2, f / (SR / 2)); return lfilter(b, a, x)
def hp(x, f): b, a = butter(2, f / (SR / 2), "high"); return lfilter(b, a, x)

def pop(f0=350, f1=1100):
    t = tt(0.14); f = f0 + (f1 - f0) * np.clip(t / 0.04, 0, 1)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 35)
def bell(f, d=2.5, bright=1.0):
    t = tt(d); s = 0
    for k, (m, a, dec) in enumerate([(1, 1, 2.2), (2.0, .5, 3), (2.76, .45 * bright, 4), (5.4, .25 * bright, 6), (8.93, .12 * bright, 9)]):
        s = s + a * np.sin(2 * np.pi * f * m * t + k) * np.exp(-t * dec)
    return s * np.minimum(1, t / 0.002) / 2
def drum(f0=160, f1=55, d=0.6, slap=0.4):  # thavil-ish low drum
    t = tt(d); f = f1 + (f0 - f1) * np.exp(-t * 25)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6)
    n = bp(rng.normal(0, 1, len(t)), 800, 4000) * np.exp(-t * 60) * slap
    return body + n
def tick(f=2200):  # mridangam-style 'ta'
    t = tt(0.12)
    return (np.sin(2 * np.pi * f * t) * .5 + bp(rng.normal(0, 1, len(t)), 2000, 6000)) * np.exp(-t * 45)
def swoosh(d=0.35, up=True):
    t = tt(d); n = rng.normal(0, 1, len(t)); env = np.sin(np.pi * t / d) ** 2
    return bp(n, 600, 5000) * env * 0.6
def riser(d):
    t = tt(d); n = rng.normal(0, 1, len(t))
    out = np.zeros_like(n); chunk = SR // 20
    for i in range(0, len(n), chunk):
        fc = 300 + 5000 * (i / len(n)) ** 2
        out[i:i + chunk] = bp(n[max(0, i - 2000):i + chunk], fc * .7, min(fc * 1.4, 20000))[-len(n[i:i + chunk]):]
    return out * (t / d) ** 2 * 0.5
def match():
    t = tt(0.35); n = rng.normal(0, 1, len(t))
    scratch = hp(n, 2500) * np.exp(-t * 14) * (1 + np.sin(t * 300))
    whoosh = bp(n, 200, 1500) * np.exp(-((t - .2) / .08) ** 2)
    return scratch * .5 + whoosh * .9
def sparkle(d=0.9, dens=18):
    out = np.zeros(int(d * SR))
    for _ in range(dens):
        f = rng.uniform(3000, 7000); st = int(rng.uniform(0, d - .15) * SR); t = tt(0.15)
        out[st:st + len(t)] += np.sin(2 * np.pi * f * t) * np.exp(-t * 30) * rng.uniform(.2, .5)
    return out
def boing():
    t = tt(0.5); f = 180 + 120 * np.exp(-t * 4) * (1 + 0.5 * np.sin(2 * np.pi * 14 * t))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6)
def rustle():
    t = tt(0.3); n = rng.normal(0, 1, len(t))
    return bp(n, 1500, 7000) * np.abs(np.sin(t * 60)) * np.exp(-t * 8) * .6
def whistle(d=0.55):
    t = tt(d); f = 900 + 1600 * t / d
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / d) * 0.25 + hp(rng.normal(0, 1, len(t)), 3000) * .05
def burst():
    t = tt(1.6); n = rng.normal(0, 1, len(t))
    boom = lp(n, 400) * np.exp(-t * 7) * 3
    crackle = np.zeros(len(t))
    for _ in range(90):
        i = int(rng.uniform(0.1, 1.4) * SR); crackle[i:i + 200] += rng.normal(0, 1, 200) * np.exp(-np.arange(200) / 40)
    return boom + hp(crackle, 1500) * .7

PENTA = [523.25, 587.33, 659.25, 783.99, 880.0, 1046.5]

# ---- drone bed (tanpura-style, C & G), the "sound healing" layer
t = tt(D.DUR)
drone = 0
for f, a in [(130.81, 1), (196.0, .7), (261.63, .6), (65.41, .5)]:
    for h in range(1, 7):
        drone = drone + a / h ** 1.3 * np.sin(2 * np.pi * f * h * t * (1 + .0007 * np.sin(t * .5 + h)))
env = np.interp(t, [0, 1.9, 2.0, 2.6, 14.0, 15.0], [0.12, 0.25, 0.25, 0.55, 0.55, 0.0])
drone = lp(drone, 2500) * env * 0.09
L += drone; R += drone

# ---- hook (0-2s)
put(riser(1.9), 0.05, 0.5)
for tm, f in [(0.25, (300, 900)), (0.65, (330, 1000)), (1.05, (380, 1200))]: put(pop(*f), tm, .6)
put(match(), 1.5, .9)
put(swoosh(.3), 1.72, .7)
# ---- BLOOM at 2.0
put(drum(140, 40, 1.4, .6), 2.0, 1.3)
put(bell(523.25, 3.0), 2.0, .5, -.3); put(bell(783.99, 3.0), 2.0, .4, .3)
put(sparkle(1.2, 30), 2.0, .6)
put(pop(), 2.0, .5)
put(drum(180, 50, .7), 2.5, 1.1); put(tick(), 2.5, .4)
put(swoosh(.3), 2.95, .6)
# toran drop
put(rustle(), 2.2, .5)
# ---- person pops in
put(boing(), 3.0, .7); put(rustle(), 3.05, .6)
# ---- diyas light (ascending bells)
for k, tm in enumerate(D.DIYA_LIT):
    put(bell(PENTA[k], 1.8), tm, .45, [-.6, -.3, .3, .6, 0][k])
    put(swoosh(.15), tm - .04, .25)
# ---- groove 3.5 → 12.5 at 120 BPM: dhum on beats, ta on off-beats
b = 3.5
while b < 12.5:
    put(drum(150, 60, .5, .25), b, .45)
    put(tick(2400 if int(b * 2) % 4 else 1900), b + .25, .22, .25)
    b += .5
# ---- kinetic text hits
for key, y, t0, t1, st in D.TEXT:
    if t0 < 2.0 or key in ("can", "every"): continue
    if st == "slam":
        put(drum(200, 45, 1.0, .7), t0, 1.2); put(sparkle(.6, 10), t0, .4)
    elif st == "drop":
        put(swoosh(.25), t0 - .05, .5); put(drum(170, 70, .4, .5), t0 + .3, .8)
    elif st == "pop":
        put(pop(400, 1300), t0, .5)
    else:
        put(sparkle(.5, 8), t0, .4)
    if t1 < 20: put(swoosh(.25), t1, .35)
# Tamil greeting = temple bell
put(bell(440, 3.5, 1.3), 6.5, .7, -.2); put(bell(660, 3.0), 7.0, .5, .2)
# wish words twinkle
for tm in (9.25, 9.75): put(sparkle(.6, 10), tm, .5)
# person hops
for th, nx in D.HOP[:-1]: put(boing(), th, .35); put(rustle(), th + .38, .3)
# ---- fireworks
for (ft, fx, fy, kind) in D.FIREWORKS:
    pan = (fx - 540) / 540
    put(whistle(), ft - .55, .5, pan); put(burst(), ft, .7, pan)
# ---- outro: sign-off shimmer + final bell chord
put(sparkle(1.4, 28), 13.75, .5)
for f, p in [(523.25, -.4), (659.25, 0), (783.99, .4), (1046.5, 0)]: put(bell(f, 1.2), 14.0, .3, p)

# ---- master: normalise, soft clip, fade
mix = np.stack([L, R], 1)
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(mix * 1.6) / np.tanh(1.6) * 0.9
fade = np.ones(N); fade[-int(.4 * SR):] = np.linspace(1, 0, int(.4 * SR)); mix *= fade[:, None]
with wave.open(os.path.join(D.S, "audio.wav"), "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("audio ok")
