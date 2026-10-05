"""Voiceover (espeak-ng) + original 120 BPM soundtrack + SFX for the F1 stop-motion short."""
import numpy as np, wave, subprocess, json, os
from scipy.signal import butter, lfilter

SR = 44100
DUR = 90.0
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{OUT}/vo", exist_ok=True)

# (scene_start, window_end, [sentences])  -- spoken text is phonetic-friendly for espeak
SCRIPT = [
    (0.1, 3.1, ["Seventy six years of F1.", "In ninety seconds!"]),
    (3.2, 12.0, ["Nineteen fifty, Silverstone.", "The first world championship race.",
                 "Front engine, skinny tyres, no seatbelts.", "Fangio wins five titles."]),
    (12.2, 20.0, ["Then Cooper moves the engine to the back.",
                  "By sixty eight, wings and sponsor colours arrive."]),
    (20.2, 28.0, ["The seventies get wild lah.", "Ground effect sucks the car onto the road.",
                  "Tyrrell even races with six wheels!"]),
    (28.2, 36.0, ["The eighties. Turbo era.", "Over one thousand horsepower in qualifying.",
                  "Senna versus Prost. Legendary."]),
    (36.4, 42.0, ["Imola, ninety four.", "A dark weekend that made safety number one."]),
    (42.1, 46.0, ["Ninety nine, F1 arrives in Malaysia.", "Sepang!"]),
    (46.2, 54.0, ["The two thousands.", "Schumacher and Ferrari win five titles in a row.",
                  "Seven in total. Wah, unreal."]),
    (54.2, 64.0, ["Twenty fourteen, the hybrid era begins.", "Hamilton matches seven titles.",
                  "And the halo starts saving lives."]),
    (64.2, 72.0, ["Twenty twenty one. A last lap title decider in Abu Dhabi.",
                  "Then ground effect returns. Verstappen rules."]),
    (72.2, 84.0, ["And now, twenty twenty six.", "Smaller, lighter cars.", "Half the power is electric.",
                  "One hundred percent sustainable fuel.", "Moving wings.", "And eleven teams on the grid."]),
    (84.2, 89.8, ["Which era is your favourite? Comment lah.", "Follow, vee liam soon, for part two!"]),
]
# caption text (what viewers read) mirrors the VO but with proper spelling
CAPTIONS = [
    ["76 years of F1.", "In 90 seconds!"],
    ["1950, Silverstone.", "The first World Championship race.", "Front engine, skinny tyres, no seatbelts.", "Fangio wins 5 titles."],
    ["Then Cooper moves the engine to the back.", "By '68, wings and sponsor colours arrive."],
    ["The '70s get wild lah.", "Ground effect sucks the car onto the road.", "Tyrrell even races with 6 wheels!"],
    ["The '80s. Turbo era.", "Over 1,000 hp in qualifying.", "Senna vs Prost. Legendary."],
    ["Imola, 1994.", "A dark weekend that made safety No. 1."],
    ["1999, F1 arrives in Malaysia.", "SEPANG!"],
    ["The 2000s.", "Schumacher & Ferrari win 5 titles in a row.", "7 in total. Wah, unreal."],
    ["2014, the hybrid era begins.", "Hamilton matches 7 titles.", "And the halo starts saving lives."],
    ["2021. A last-lap title decider in Abu Dhabi.", "Then ground effect returns. Verstappen rules."],
    ["And now, 2026.", "Smaller, lighter cars.", "Half the power is electric.", "100% sustainable fuel.", "Moving wings.", "And 11 teams on the grid."],
    ["Which era is your favourite? Comment lah.", "Follow @vliamsoon for Part 2!"],
]


def read_wav(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    # trim silence
    nz = np.where(np.abs(a) > 0.01)[0]
    if len(nz):
        a = a[max(0, nz[0] - 200): nz[-1] + 400]
    n = int(len(a) * SR / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def tts(text, speed, path):
    subprocess.run(["espeak-ng", "-v", "en-gb+m3", "-s", str(speed), "-p", "38", "-g", "1", "-w", path, text],
                   check=True)
    return read_wav(path)


def lp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "low"); return lfilter(b, a, x)


def hp(x, fc, order=2):
    b, a = butter(order, fc / (SR / 2), "high"); return lfilter(b, a, x)


def bp(x, lo, hi, order=2):
    b, a = butter(order, [lo / (SR / 2), hi / (SR / 2)], "band"); return lfilter(b, a, x)


# ---------------- voiceover ----------------
N = int(DUR * SR)
voice = np.zeros(N, np.float32)
timing = []
for si, (start, end, sents) in enumerate(SCRIPT):
    gap = 0.12
    speed = 168
    for _ in range(8):
        clips = [tts(s, speed, f"{OUT}/vo/s{si}_{k}.wav") for k, s in enumerate(sents)]
        total = sum(len(c) for c in clips) / SR + gap * (len(clips) - 1)
        if total <= (end - start) - 0.15 or speed >= 205:
            break
        speed += 7
    t = start
    sent_t = []
    for k, c in enumerate(clips):
        i = int(t * SR)
        seg = c[: max(0, N - i)]
        voice[i:i + len(seg)] += seg
        sent_t.append([round(t, 3), round(t + len(c) / SR, 3), CAPTIONS[si][k]])
        t += len(c) / SR + gap
    timing.append({"scene": si, "speed": speed, "sentences": sent_t})
    print(f"scene {si}: speed {speed}, vo {total:.2f}s / window {end - start:.2f}s")

# voice polish: warm it up, light compression
voice = hp(voice, 90)
voice = voice + 0.25 * lp(voice, 300)
pk = np.max(np.abs(voice)) + 1e-9
voice = voice / pk
voice = np.tanh(voice * 1.6) / np.tanh(1.6)

json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1)

# ---------------- music: 120 BPM, A minor ----------------
BPM = 120; BEAT = 60 / BPM; BAR = BEAT * 4
t_all = np.arange(N) / SR
rng = np.random.default_rng(7)
drums = np.zeros(N); bass = np.zeros(N); arp = np.zeros(N); pad = np.zeros(N); lead = np.zeros(N)


def note_hz(n):  # midi
    return 440 * 2 ** ((n - 69) / 12)


def add(buf, sig, t0):
    i = int(t0 * SR)
    if i >= N: return
    s = sig[: N - i]; buf[i:i + len(s)] += s


def env(n, a=0.005, d=0.2):
    tt = np.arange(n) / SR
    return np.minimum(1, tt / a) * np.exp(-tt / d)


def kick():
    n = int(0.35 * SR); tt = np.arange(n) / SR
    f = 45 + 110 * np.exp(-tt / 0.04)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.13) * 1.0


def snare():
    n = int(0.25 * SR); tt = np.arange(n) / SR
    nz = bp(rng.standard_normal(n), 1500, 7000) * np.exp(-tt / 0.07)
    tone = np.sin(2 * np.pi * 190 * tt) * np.exp(-tt / 0.05)
    return 0.55 * nz + 0.35 * tone


def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR); tt = np.arange(n) / SR
    return hp(rng.standard_normal(n), 7000) * np.exp(-tt / (0.09 if open_ else 0.018)) * 0.25


def crash():
    n = int(1.6 * SR); tt = np.arange(n) / SR
    return hp(rng.standard_normal(n), 4000) * np.exp(-tt / 0.5) * 0.3


def saw(f, n):
    ph = (np.arange(n) * f / SR) % 1.0
    return 2 * ph - 1


def sq(f, n, duty=0.5):
    ph = (np.arange(n) * f / SR) % 1.0
    return np.where(ph < duty, 1.0, -1.0)


K, S_, HAT, OH, CR = kick(), snare(), hat(), hat(True), crash()
prog = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]  # Am F C G
roots = [45, 41, 36, 43]

SOFT = (36.0, 42.0)   # respectful breakdown (Imola)
nbars = int(DUR / BAR) + 1
for b in range(nbars):
    t0 = b * BAR
    ch = prog[b % 4]; root = roots[b % 4]
    soft = SOFT[0] <= t0 < SOFT[1]
    final = t0 >= 72
    cta = t0 >= 84
    # pad (always)
    n = int(BAR * SR)
    p = sum(saw(note_hz(m) * d, n) for m in ch for d in (0.997, 1.003))
    p = p * np.minimum(1, np.arange(n) / (0.08 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.05 * SR))
    add(pad, p * (0.05 if not soft else 0.08), t0)
    if not soft:
        for q in range(4):
            tb = t0 + q * BEAT
            add(drums, K, tb)
            if q in (1, 3): add(drums, S_, tb)
            add(drums, HAT, tb + BEAT / 2)
            if final: add(drums, HAT * 0.6, tb + BEAT / 4); add(drums, HAT * 0.6, tb + 3 * BEAT / 4)
        if b % 4 == 3: add(drums, OH, t0 + 3.5 * BEAT)
        # bass 8ths
        for e in range(8):
            n = int(BEAT / 2 * SR * 0.9)
            f = note_hz(root + (12 if e % 4 == 3 else 0))
            add(bass, saw(f, n) * env(n, 0.003, 0.18) * 0.5, t0 + e * BEAT / 2)
    # arp 16ths (soft section: slower, gentler)
    step = BEAT / 4 if not soft else BEAT / 2
    seq = ch + [c + 12 for c in ch] + [ch[1] + 12, ch[2]]
    for k in range(int(BAR / step)):
        n = int(step * SR)
        m = seq[k % len(seq)] + 12
        a = (sq(note_hz(m), n, 0.3) * 0.5 + np.sin(2 * np.pi * note_hz(m) * np.arange(n) / SR)) * env(n, 0.002, 0.09 if not soft else 0.3)
        add(arp, a * (0.09 if not soft else 0.07), t0 + k * step)
    # lead hook in the 2026 + CTA section
    if final:
        motif = [(0, 76, 1), (1, 74, .5), (1.5, 72, .5), (2, 71, 1), (3, 72, 1)]
        tr = [0, -4, -9, -2][b % 4] if False else 0
        for off, m, ln in motif:
            n = int(ln * BEAT * SR)
            mm = m + {0: 0, 1: -3, 2: -5, 3: -2}[b % 4]
            sig = (saw(note_hz(mm), n) + 0.5 * saw(note_hz(mm) * 1.005, n)) * env(n, 0.01, 0.35)
            add(lead, sig * 0.06, t0 + off * BEAT)

# crashes on section downbeats
for tcr in [0, 3, 12, 20, 28, 42, 46, 54, 64, 72, 84]:
    add(drums, CR, tcr)
# riser into the Sepang drop and the 2026 drop
for t_drop, ln in [(42.0, 2.0), (72.0, 2.0)]:
    n = int(ln * SR); tt = np.arange(n) / SR
    r = hp(rng.standard_normal(n), 2000) * (tt / ln) ** 2 * 0.25
    add(drums, r, t_drop - ln)

arp = lp(arp, 4500); bass = lp(bass, 900); pad = lp(pad, 2200); lead = lp(lead, 3500)
music = drums * 0.8 + bass + arp + pad + lead
music[: int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))
fade = int(0.8 * SR); music[-fade:] *= np.linspace(1, 0, fade)

# ---------------- SFX: whooshes + engine pass-bys ----------------
sfx = np.zeros(N)
for tw in [3, 12, 20, 28, 42, 46, 54, 64, 72, 84]:
    n = int(0.5 * SR); tt = np.arange(n) / SR
    w = bp(rng.standard_normal(n), 400, 3000) * np.sin(np.pi * tt / 0.5) ** 2
    add(sfx, w * 0.5, tw - 0.25)


def engine_pass(t0, ln=1.4, f0=150, f1=260, gain=0.35, turbo=False):
    n = int(ln * SR); tt = np.arange(n) / SR
    u = tt / ln
    f = f0 + (f1 - f0) * u - 80 * np.clip(u - 0.55, 0, 1)  # doppler drop
    ph = np.cumsum(f) / SR
    s = saw(1, n) * 0
    s = (2 * (ph % 1) - 1) + 0.5 * (2 * ((ph * 2.01) % 1) - 1)
    s = lp(s, 2200) * np.sin(np.pi * u) ** 1.5
    if turbo:
        s += np.sin(2 * np.pi * np.cumsum(f * 9) / SR) * 0.15 * np.sin(np.pi * u)
    add(sfx, s * gain, t0)


engine_pass(0.4, 1.6, 180, 330, 0.45)       # hook smash run
engine_pass(9.5, 1.6, 110, 190, 0.3)        # 1950 drive off
engine_pass(18.0, 1.5, 140, 240, 0.3)
engine_pass(26.2, 1.5, 150, 260, 0.3)
engine_pass(28.0, 1.4, 170, 300, 0.45, turbo=True)
engine_pass(43.4, 1.8, 190, 330, 0.4)       # Sepang pass
engine_pass(52.3, 1.4, 200, 360, 0.35)
engine_pass(64.3, 3.0, 190, 300, 0.3)
engine_pass(84.6, 1.8, 200, 340, 0.35, turbo=True)
# impact for the hook smash
n = int(0.6 * SR); tt = np.arange(n) / SR
add(sfx, (lp(rng.standard_normal(n), 1200) * 1.2 + np.sin(2 * np.pi * 60 * tt)) * np.exp(-tt / 0.15) * 0.8, 1.0)

# ---------------- mix with ducking ----------------
e = lp(np.abs(voice), 8, 1)
e = np.clip(e / (np.percentile(e[e > 1e-4], 60) + 1e-9), 0, 1)
duck = 1 - 0.55 * e
mix = music * 0.55 * duck + sfx * 0.45 * (1 - 0.3 * e) + voice * 0.95
mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.89


def write(path, x):
    x = np.clip(x, -1, 1)
    st = (np.stack([x, x], 1) * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(st.tobytes())


write(f"{OUT}/mix.wav", mix)
write(f"{OUT}/music_only.wav", (music * 0.55 + sfx * 0.45) / (np.max(np.abs(music * 0.55 + sfx * 0.45)) + 1e-9) * 0.89)
write(f"{OUT}/voice_only.wav", voice * 0.9)
print("audio done")
