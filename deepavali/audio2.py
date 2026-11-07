"""Indian-style score for v2: tanpura, sitar (Mohanam raga), tabla keherwa, bansuri, shankh, ghungroo, temple bells."""
import numpy as np, wave, os
from scipy.signal import lfilter, butter
import deepavali2 as V
from audio import tt, bp, lp, hp, pop, bell, swoosh, riser, match, sparkle, whistle, burst  # reuse SFX
import audio as A0

SR = 44100; N = int(SR * V.DUR)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(11)
def put(sig, t, gain=1.0, pan=0.0):
    i = int(t * SR); j = min(N, i + len(sig))
    if i >= N or j <= i: return
    s = sig[:j - i] * gain
    L[i:j] += s * np.sqrt((1 - pan) / 2); R[i:j] += s * np.sqrt((1 + pan) / 2)

SA = 261.63  # C
def sw(n): return SA * 2 ** (n / 12)
# Mohanam swaras: S R G P D  (0 2 4 7 9)
MO = {"S": 0, "R": 2, "G": 4, "P": 7, "D": 9, "S'": 12, "R'": 14, "G'": 16, "P.": -5, "D.": -3}

# ---------------------------------------------------------- instruments
def sitar(f, d=1.2, glide_to=None, glide_at=0.35):
    t = tt(d)
    fr = np.full(len(t), f)
    if glide_to:
        g = np.clip((t - glide_at) / 0.18, 0, 1); g = g * g * (3 - 2 * g)
        fr = f * (glide_to / f) ** g
    ph = 2 * np.pi * np.cumsum(fr) / SR
    c = 4 + 14 * np.exp(-t * 2.5)  # jawari: moving formant over harmonics = buzz
    out = np.zeros(len(t))
    for h in range(1, 28):
        w = (1 / h ** .75) * (1 + 1.8 * np.exp(-((h - c) / 3.0) ** 2))
        out += w * np.sin(h * ph + h) * np.exp(-t * (1.2 + .22 * h))
    out += hp(rng.normal(0, 1, len(t)), 3000) * np.exp(-t * 120) * .6   # mizrab pluck
    out *= np.minimum(1, t / .003)
    return out / 6

def tarb(f, d=2.0):  # sympathetic shimmer
    t = tt(d); return sum(np.sin(2 * np.pi * f * m * t) * np.exp(-t * 2.5) / m for m in (2, 3, 4)) * .15

def bansuri(notes, start):
    """notes: list of (swara, dur). legato flute with vibrato + breath"""
    total = sum(d for _, d in notes); t = tt(total)
    fr = np.zeros(len(t)); acc = 0; prev = None
    for sv, d in notes:
        i0, i1 = int(acc * SR), int((acc + d) * SR)
        f = sw(MO[sv] + 12)
        if prev:
            g = np.clip(np.arange(i1 - i0) / (0.08 * SR), 0, 1)
            fr[i0:i1] = prev * (f / prev) ** g
        else: fr[i0:i1] = f
        prev = f; acc += d
    vib = 1 + 0.007 * np.sin(2 * np.pi * 5.3 * t) * np.clip((t - .3) / .4, 0, 1)
    ph = 2 * np.pi * np.cumsum(fr * vib) / SR
    env = np.clip(t / .12, 0, 1) * np.clip((total - t) / .3, 0, 1)
    tone = np.sin(ph) + .25 * np.sin(2 * ph) + .08 * np.sin(3 * ph)
    breath = bp(rng.normal(0, 1, len(t)), 1500, 5000) * .12
    put((tone + breath) * env * .32, start, 1.0, .15)

def tanpura_pluck(f, d=3.0):
    t = tt(d); out = np.zeros(len(t))
    c = 3 + 20 * np.clip(t / 2.5, 0, 1)  # jawari bloom rising through harmonics
    for h in range(1, 30):
        w = (1 / h ** .9) * (1 + 2.5 * np.exp(-((h - c) / 2.5) ** 2))
        out += w * np.sin(2 * np.pi * f * h * t * (1 + .0004 * h)) * np.exp(-t * (.6 + .05 * h))
    return out * np.minimum(1, t / .01) / 8

# tabla
def bayan(f0=85, bend=1.35, d=.5):
    t = tt(d); f = f0 * (1 + (bend - 1) * np.clip(t / .18, 0, 1))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) + lp(rng.normal(0, 1, len(t)), 300) * np.exp(-t * 40) * .3
def dayan(f=SA * 2, ring=1.0, d=.35):
    t = tt(d)
    s = (np.sin(2 * np.pi * f * t) + .5 * np.sin(2 * np.pi * f * 2.0 * t) + .3 * np.sin(2 * np.pi * f * 3.0 * t)) * np.exp(-t * (14 / ring))
    return s + bp(rng.normal(0, 1, len(t)), 2500, 8000) * np.exp(-t * 90) * .5
def mixs(*xs):
    n = max(len(x) for x in xs); o = np.zeros(n)
    for x in xs: o[:len(x)] += x
    return o
def ke(): t = tt(.08); return bp(rng.normal(0, 1, len(t)), 300, 1500) * np.exp(-t * 60)
STROKE = {
    "dha": lambda: mixs(dayan() * .8, bayan() * .9), "dhin": lambda: mixs(dayan(ring=.6) * .6, bayan() * .9),
    "na": lambda: dayan() * .8, "tin": lambda: dayan(SA * 2, .5) * .55, "ge": lambda: bayan(),
    "ke": lambda: ke() * .9, "ta": lambda: dayan(SA * 4, .4) * .6, "tit": lambda: dayan(SA * 4, .25) * .45,
}
def tabla(seq, start, step, gain=1.0):
    for i, s in enumerate(seq):
        if s != "-": put(STROKE[s](), start + i * step, gain, -.1 if s in ("ge", "ke") else .1)

def shankh(d=1.8):  # conch
    t = tt(d); f = 233 * (1 + .03 * np.clip(t / .6, 0, 1))
    ph = 2 * np.pi * np.cumsum(f * (1 + .004 * np.sin(2 * np.pi * 4.5 * t))) / SR
    tone = sum(np.sin(h * ph) / h ** .6 for h in range(1, 14))
    env = np.clip(t / .35, 0, 1) * np.clip((d - t) / .5, 0, 1)
    return bp(tone, 180, 3500) * env * .25 + bp(rng.normal(0, 1, len(t)), 400, 2000) * env * .03

def ghungroo(d=.5, n=14):
    out = np.zeros(int(d * SR))
    for _ in range(n):
        st = int(rng.uniform(0, d - .12) * SR); t = tt(.12); f = rng.uniform(4200, 7800)
        out[st:st + len(t)] += (np.sin(2 * np.pi * f * t) + .5 * np.sin(2 * np.pi * f * 1.48 * t)) * np.exp(-t * 35) * rng.uniform(.3, .6)
    return out

def flutter(d):  # peacock wing beats (12 fps stop-motion => flaps at 1.5 Hz cycle of 8 frames)
    t = tt(d); env = np.abs(np.sin(np.pi * 1.5 * t)) ** 4
    return bp(rng.normal(0, 1, len(t)), 300, 1400) * env * .35

def peacock_call():
    out = []
    for f0, f1, d in [(820, 1250, .16), (1250, 900, .22)]:
        t = tt(d); f = f0 + (f1 - f0) * t / d
        ph = 2 * np.pi * np.cumsum(f) / SR
        s = sum(np.sin(h * ph) / h for h in range(1, 6)) * np.sin(np.pi * t / d) ** .5
        out.append(bp(s, 600, 4000)); out.append(np.zeros(int(.04 * SR)))
    return np.concatenate(out) * .4

# ---------------------------------------------------------- score
# tanpura: P S' S' S (pluck every 0.5s) from 0 to end
pat = [sw(-5), sw(12) / 2 * 2, sw(0) * 2, sw(0)]
k = 0; tp = 0.0
while tp < V.DUR - .5:
    g = .35 if tp < 2 else .6
    put(tanpura_pluck(pat[k % 4] / 2), tp, g * .5, [-.4, .2, .4, -.2][k % 4]); k += 1; tp += .5

# --- hook 0–2s (keeps the suspense)
put(riser(1.9), .05, .45)
for tm, n in [(.25, "S"), (.65, "G"), (1.05, "P")]:
    put(sitar(sw(MO[n]), .9), tm, .7, .1); put(pop(300, 900), tm, .25)
put(match(), 1.5, .8); put(swoosh(.3), 1.72, .6)
# --- bloom 2.0: shankh + big dha + bells
put(shankh(1.8), 1.95, 1.0)
put(STROKE["dha"]() * 1.5, 2.0, 1.0); put(A0.drum(140, 40, 1.4, .5), 2.0, .9)
put(bell(523.25, 3.0), 2.0, .4, -.3); put(bell(783.99, 3.0), 2.0, .3, .3)
put(sparkle(1.2, 26), 2.0, .5)
put(sitar(sw(12), 1.0), 2.0, .6)
tabla(["dha", "-", "tit", "tit", "dha"], 2.25, .0625, 1.0)   # little flourish into EVERYTHING
put(A0.drum(180, 50, .7), 2.5, .8)
put(swoosh(.3), 2.95, .5)
# --- 3.0: peacock enters, sitar run, tabla keherwa begins
put(peacock_call(), 3.15, .55, -.5)
put(flutter(2.3), 3.0, .8, 0)
run = ["S", "R", "G", "P", "D", "S'"]
for i, n in enumerate(run): put(sitar(sw(MO[n])), 3.0 + i * .125, .75, .1)
put(sitar(sw(MO["S'"]), 1.5, glide_to=sw(MO["D"]), glide_at=.5), 3.75, .75, .1)
KEHERWA = ["dha", "ge", "na", "tin", "na", "ke", "dhin", "na"]
FILL = ["dha", "tit", "tit", "dha", "tit", "tit", "dha", "-"]
tabla(KEHERWA[:4], 3.0, .25, .85); tabla(FILL, 4.0, .125, .8)
tabla(KEHERWA, 5.0, .25, .85); tabla(KEHERWA[:4], 7.0, .25, .85); tabla(FILL * 2, 8.0, .0625 * 2 / 2 * 1, .85) if False else tabla(FILL, 8.0, .125, .9)
tabla(KEHERWA, 9.0, .25, .75); tabla(KEHERWA[:6], 11.0, .25, .75)
# lamps lighting: temple bells in Mohanam
for i, tm in enumerate(V.DIYA_LIT): put(bell(sw(MO[["S'", "R'", "G'", "P"][i]]), 1.6), tm, .35, [-.6, -.3, .3, .6][i])
put(bell(sw(MO["S'"]), 1.6), 1.75, .3)
# fan opening 4.8: ghungroo + shimmer
put(ghungroo(.7, 22), 4.8, .6); put(sparkle(.8, 16), 4.85, .4); put(swoosh(.5), 4.75, .45)
# --- greeting 5–9: sitar melody (Mohanam phrase), hits on text
mel = [(5.0, "G"), (5.25, "P"), (5.5, "S'"), (6.0, "D"), (6.25, "P"), (6.5, "G"), (6.75, "R"),
       (7.0, "S"), (7.25, "R"), (7.5, "G"), (7.75, "P"), (8.0, "D"), (8.25, "S'"), (8.5, "R'"), (8.75, "S'")]
for tm, n in mel: put(sitar(sw(MO[n]), 1.0), tm, .55, .15); put(tarb(sw(MO[n])), tm, .4)
put(sitar(sw(MO["G"]), 1.4, glide_to=sw(MO["P"]), glide_at=.2), 7.0, .4, -.2)
put(bell(440, 3.5, 1.3), 6.5, .55, -.2); put(bell(660, 3.0), 7.0, .4, .2)
for (t0, t1, pts, sc, beh) in V.FLIGHTS[1:]:
    put(flutter(min(t1, 15) - t0), t0, .45 * sc, -.3 if pts[0][0] < 0 else .3)
put(peacock_call(), 6.6, .3, -.6); put(peacock_call(), 12.5, .3, .6)
# --- doll entrance 9.0: shankh + ghungroo + dha; namaste bow bell 9.7
put(shankh(1.4), 8.85, .7)
put(STROKE["dha"]() * 1.4, 9.0, 1.0); put(ghungroo(.6, 20), 9.0, .7); put(A0.boing(), 9.0, .25)
put(bell(sw(MO["S'"]), 2.5), 9.7, .45)
# bansuri (sacred) over the wish 9.2–12.4
bansuri([("G", .5), ("P", .5), ("D", .75), ("P", .25), ("G", .5), ("R", .5), ("G", .5), ("S", .25)], 9.2)
for th, nx in V.DOLL_HOP: put(ghungroo(.35, 9), th, .5); put(A0.rustle(), th + .38, .25)
# --- outro 12.5–15: jhala strums + final
for i in range(16):
    n = ["S", "S'", "S'", "S'"][i % 4] if i % 8 < 4 else ["P", "S'", "S'", "S'"][i % 4]
    put(sitar(sw(MO[n]) * (1 if i % 4 == 0 else 2), .4), 12.5 + i * .0625 * 2, .35 if i % 4 else .55, .1)
tabla(["dha", "-", "dha", "-", "tit", "tit", "dha", "-", "na", "tin", "na", "ke", "dha", "-", "-", "-"], 12.5, .125, .85)
tabla(["dha", "tit", "tit", "dha", "-", "dha", "tit", "tit", "dha", "-", "dha", "tit", "tit", "dha"], 14.0 - 13 * .07, .07, .9)  # tihai lands on 14.0
put(sitar(sw(MO["S'"]), 1.6, glide_to=sw(MO["S"]) * 2, glide_at=2), 14.0, .8)
put(sitar(sw(MO["S"]), 1.6), 14.0, .6)
for f, p in [(523.25, -.4), (659.25, 0), (783.99, .4)]: put(bell(f, 1.2), 14.0, .25, p)
put(sparkle(1.3, 26), 13.75, .45)
# kinetic text accents (Indian strokes)
for key, y, t0, t1, st in V.TEXT:
    if t0 < 5: continue
    if st == "slam": put(STROKE["dha"]() * 1.2, t0, .9); put(A0.drum(200, 45, .9, .5), t0, .8)
    elif st == "drop": put(swoosh(.25), t0 - .05, .4); put(bayan(70, 1.6, .6), t0 + .3, 1.0)
    elif st == "pop": put(STROKE["na"](), t0, .5)
    else: put(sparkle(.5, 8), t0, .35)
    if t1 < 20: put(swoosh(.25), t1, .3)
for (ft, fx, fy, kind) in V.FIREWORKS:
    pan = (fx - 540) / 540
    put(whistle(), ft - .55, .4, pan); put(burst(), ft, .55, pan)

mix = np.stack([L, R], 1)
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(mix * 1.8) / np.tanh(1.8) * .92
fade = np.ones(N); fade[-int(.5 * SR):] = np.linspace(1, 0, int(.5 * SR)); mix *= fade[:, None]
with wave.open(os.path.join(V.S, "audio2.wav"), "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("audio2 ok")
