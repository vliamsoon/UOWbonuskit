"""Stop-motion style F1 history short: modular 3D blocks, 9:16, 12 fps."""
import math, random, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from multiprocessing import Pool

OUT = os.path.dirname(os.path.abspath(__file__))
W, H, SS = 1080, 1920, 2           # output size, supersample factor
FPS, DUR = 12, 90
FRAMES = FPS * DUR
TIMING = json.load(open(f"{OUT}/timing.json"))
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_fonts = {}


def font(sz):
    k = int(sz * SS)
    if k not in _fonts: _fonts[k] = ImageFont.truetype(FB, k)
    return _fonts[k]


# ---------------------------------------------------------------- projection
TH, EL = math.radians(24), math.radians(32)
cT, sT, cE, sE = math.cos(TH), math.sin(TH), math.cos(EL), math.sin(EL)
SCALE = 57.0
CX, CY = 540.0, 1010.0
JIT = [0, 0]


def P(x, y, z):
    sx = CX + JIT[0] + SCALE * (x * cT - y * sT)
    sy = CY + JIT[1] + SCALE * ((x * sT + y * cT) * sE - z * cE)
    return (sx * SS, sy * SS)


def shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


def box(x0, y0, z0, x1, y1, z1, c):
    return (min(x0, x1), min(y0, y1), min(z0, z1), max(x0, x1), max(y0, y1), max(z0, z1), c)


def move(B, dx=0, dy=0, dz=0):
    return [(b[0] + dx, b[1] + dy, b[2] + dz, b[3] + dx, b[4] + dy, b[5] + dz, b[6]) for b in B]


def scale(B, s, sz=None):
    sz = s if sz is None else sz
    return [(b[0] * s, b[1] * s, b[2] * sz, b[3] * s, b[4] * s, b[5] * sz, b[6]) for b in B]


def rot90(B):  # nose +x -> +y
    return [box(-b[1], b[0], b[2], -b[4], b[3], b[5], b[6]) for b in B]


def recolor(B, mapping):
    return [b[:6] + (mapping.get(b[6], b[6]),) for b in B]


def depth(b):
    cx, cy, cz = (b[0] + b[3]) / 2, (b[1] + b[4]) / 2, (b[2] + b[5]) / 2
    return (cx * sT + cy * cT) * cE + cz * sE


def hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3: return pts
    def cr(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def draw_shadows(img, B, alpha=70):
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    for b in B:
        if b[5] <= 0.01: continue
        pts = []
        for x in (b[0], b[3]):
            for y in (b[1], b[4]):
                for z in (max(0, b[2]), b[5]):
                    pts.append(P(x - 0.35 * z, y - 0.55 * z, 0))
        md.polygon(hull(pts), fill=255)
    dark = Image.new("RGB", img.size, (10, 10, 20))
    img.paste(dark, (0, 0), mask.point(lambda v: v * alpha // 255))


def draw_boxes(d, B, outline=True):
    for b in sorted(B, key=depth):
        x0, y0, z0, x1, y1, z1, c = b
        ol = shade(c, 0.35) if outline else None
        w = int(1.6 * SS)
        fx = [P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1)]
        fy = [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)]
        tp = [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)]
        d.polygon(fx, fill=shade(c, 0.62), outline=ol, width=w)
        d.polygon(fy, fill=shade(c, 0.80), outline=ol, width=w)
        d.polygon(tp, fill=shade(c, 1.08), outline=ol, width=w)
        # tiny bevel highlight on the top-front edge (toy plastic look)
        if (x1 - x0) * (y1 - y0) > 0.05:
            d.line([P(x0, y1, z1), P(x1, y1, z1)], fill=shade(c, 1.35), width=max(1, int(1.2 * SS)))


# ---------------------------------------------------------------- palette
DK = (28, 28, 32); TYRE = (34, 34, 38); SIL = (190, 196, 205); GOLD = (240, 190, 40)
WHT = (240, 240, 236); GLASS = (150, 205, 235); RED = (215, 30, 35)


def wheels(xs, r, w, ys, hub=SIL):
    B = []
    for x in xs:
        for s in (-1, 1):
            y = s * ys
            B.append(box(x - r, y - w / 2, 0, x + r, y + w / 2, 2 * r, TYRE))
            if s > 0: B.append(box(x - r * .45, y + w / 2, r * .55, x + r * .45, y + w / 2 + .06, r * 1.45, hub))
    return B


def car_1950(c=(200, 35, 35), helmet=(140, 90, 50)):
    return [box(-4.2, -.7, .5, 3.6, .7, 1.6, c), box(3.6, -.55, .6, 4.6, .55, 1.45, c),
            box(4.6, -.4, .75, 4.72, .4, 1.3, DK), box(-5.2, -.45, .7, -4.2, .45, 1.4, c),
            box(1.2, -.5, 1.6, 3.4, .5, 1.75, shade(c, .85)),
            box(-2.2, -.45, 1.6, -.8, .45, 1.66, DK), box(-.9, -.4, 1.6, -.75, .4, 1.95, GLASS),
            box(.2, .7, .8, 1.4, .74, 1.4, WHT), box(-3.5, .7, .55, 1.0, .86, .72, SIL),
            ] + wheels([-3, 3], .9, .42, 1.25) + [
            box(-2.0, -.35, 1.6, -1.2, .35, 2.0, (230, 228, 215)), box(-1.95, -.3, 2.0, -1.25, .3, 2.45, helmet)]


def car_1960(c=(20, 110, 60), helmet=(230, 230, 230), wing=None, stripe=None):
    B = [box(-3.2, -.65, .35, 3.0, .65, 1.15, c), box(3.0, -.45, .4, 4.3, .45, .95, c),
         box(4.3, -.3, .5, 4.4, .3, .85, DK),
         box(-3.7, -.5, .8, -1.9, .5, 1.4, SIL), box(-4.5, -.3, .9, -3.7, -.15, 1.05, SIL),
         box(-4.5, .15, .9, -3.7, .3, 1.05, SIL), box(-.6, -.4, 1.15, -.45, .4, 1.5, GLASS)]
    if stripe: B += [box(-3.2, -.18, 1.15, 4.3, .18, 1.2, stripe)]
    B += wheels([-2.8, 2.6], .75, .6, 1.15)
    B += [box(-1.5, -.3, 1.15, -.75, .3, 1.7, helmet)]
    if wing:
        B += [box(-3.0, -.08, 1.4, -2.8, .08, 2.4, DK), box(-3.6, -1.2, 2.4, -2.4, 1.2, 2.55, wing),
              box(3.9, -1.1, .4, 4.5, 1.1, .5, wing)]
    return B


def car_1970(c=(25, 60, 165), c2=WHT, six=False, helmet=(250, 210, 30)):
    B = [box(-2.5, -.6, .3, 2.8, .6, 1.1, c), box(2.8, -.5, .3, 4.2, .5, .8, c), box(4.2, -.4, .3, 4.8, .4, .55, c),
         box(-2.0, -1.4, .3, 1.0, -.6, .9, c), box(-2.0, .6, .3, 1.0, 1.4, .9, c),
         box(-3.8, -.5, .4, -2.5, .5, 1.1, (70, 70, 75)), box(-1.9, -.35, 1.1, -1.0, .35, 2.0, c2)]
    B += wheels([-3.0], .85, .9, 1.55)
    B += wheels([2.0, 3.4], .45, .5, 1.0) if six else wheels([3.1], .65, .6, 1.3)
    B += [box(-4.8, -1.5, .8, -3.8, -1.4, 2.3, c), box(-4.8, 1.4, .8, -3.8, 1.5, 2.3, c),
          box(-4.8, -1.4, 2.0, -3.8, 1.4, 2.25, c2), box(4.0, -1.5, .25, 4.9, 1.5, .4, c2),
          box(-1.0, -.3, 1.1, -.4, .3, 1.6, helmet)]
    return B


def car_1980(c=WHT, c2=(220, 30, 40), helmet=(250, 215, 0)):
    B = [box(-2.8, -.55, .25, 3.2, .55, .95, c), box(3.2, -.4, .25, 4.8, .4, .6, c2),
         box(-2.6, -1.35, .25, 1.2, -.55, .75, c2), box(-2.6, .55, .25, 1.2, 1.35, .75, c2),
         box(-3.6, -.5, .95, -1.2, .5, 1.25, c)]
    B += wheels([-3.3], .85, .95, 1.45) + wheels([3.2], .65, .7, 1.35)
    B += [box(-5.0, -1.5, .7, -4.0, -1.4, 2.0, c), box(-5.0, 1.4, .7, -4.0, 1.5, 2.0, c),
          box(-5.0, -1.4, 1.7, -4.0, 1.4, 2.0, c2), box(4.2, -1.5, .2, 5.1, 1.5, .32, c),
          box(-1.1, -.3, .95, -.5, .3, 1.45, helmet)]
    return B


def car_modern(c, c2, helmet=WHT, nose="high", halo=False, big_wheels=False, energy=None):
    r_f, r_r = (.8, .85) if big_wheels else (.7, .8)
    B = [box(-2.6, -.5, .3, 2.4, .5, 1.0, c)]
    if nose == "high":
        B += [box(2.4, -.3, .6, 4.6, .3, .95, c), box(4.0, -.05, .25, 4.2, .05, .6, DK)]
    else:
        B += [box(2.4, -.3, .3, 4.9, .3, .7, c)]
    B += [box(-2.6, -1.3, .25, .8, -.5, .85, c), box(-2.6, .5, .25, .8, 1.3, .85, c),
          box(-4.2, -.4, 1.0, -1.0, .4, 1.4, c), box(-1.6, -.3, 1.0, -.9, .3, 1.75, c2)]
    if energy: B += [box(-2.6, 1.3, .55, .8, 1.36, .7, energy), box(-2.6, -1.36, .55, .8, -1.3, .7, energy)]
    B += wheels([-3.4], r_r, .9, 1.45, hub=c if big_wheels else SIL) + wheels([3.0], r_f, .75, 1.4, hub=c if big_wheels else SIL)
    B += [box(4.3, -1.6, .15, 5.3, 1.6, .3, c2), box(4.3, -1.6, .15, 5.3, -1.5, .6, c), box(4.3, 1.5, .15, 5.3, 1.6, .6, c),
          box(-5.2, -1.3, .6, -4.3, -1.2, 2.0, c), box(-5.2, 1.2, .6, -4.3, 1.3, 2.0, c),
          box(-5.2, -1.2, 1.6, -4.3, 1.2, 1.9, c2),
          box(-.8, -.28, 1.0, -.2, .28, 1.5, helmet)]
    if halo:
        B += [box(-1.1, -.45, 1.55, .2, -.35, 1.65, DK), box(-1.1, .35, 1.55, .2, .45, 1.65, DK),
              box(.1, -.45, 1.55, .3, .45, 1.65, DK), box(.3, -.06, 1.0, .45, .06, 1.6, DK)]
    return B


def car_2026(c=(18, 22, 30), c2=(60, 255, 150), aero=0.0, helmet=WHT):
    B = car_modern(c, c, helmet, nose="low", halo=True, big_wheels=True, energy=c2)
    # active aero: upper flaps lift open on the straights
    B += [box(4.3, -1.5, .32 + .35 * aero, 5.1, 1.5, .42 + .35 * aero, c2),
          box(-5.15, -1.2, 1.95 + .45 * aero, -4.45, 1.2, 2.08 + .45 * aero, c2)]
    return scale(B, .92)


def track(x0=-30, x1=30, w=3.0, asphalt=(70, 72, 80), grass=(70, 150, 70), kerb=(RED, WHT), z=0):
    B = [box(x0, -w, z - .5, x1, w, z, asphalt),
         box(x0, w + .6, z - .5, x1, w + 6, z - .05, grass), box(x0, -w - 6, z - .5, x1, -w - .6, z - .05, grass)]
    k = 0; x = x0
    while x < x1:
        col = kerb[k % 2]
        B += [box(x, w, z - .5, x + 1.5, w + .6, z + .12, col), box(x, -w - .6, z - .5, x + 1.5, -w, z + .12, col)]
        x += 1.5; k += 1
    return B


def trophy(x, y, z=0, s=1.0, col=GOLD):
    return move(scale([box(-.4, -.4, 0, .4, .4, .25, (60, 50, 40)), box(-.12, -.12, .25, .12, .12, .7, col),
                       box(-.45, -.45, .7, .45, .45, 1.4, col), box(.45, -.08, .9, .62, .08, 1.25, col)], s), x, y, z)


def helmet_block(c, stripe, visor=DK):
    return [box(-1.2, -1.2, 0, 1.2, 1.2, 2.2, c), box(-1.25, -1.25, 1.0, 1.25, 1.25, 1.3, stripe),
            box(-.9, 1.2, .9, 1.2, 1.32, 1.6, visor), box(1.2, -.9, .9, 1.32, 1.1, 1.6, visor)]


# pixel digits for the hook wall
DIG = {"1": [".#.", "##.", ".#.", ".#.", "###"], "9": ["###", "#.#", "###", "..#", "###"],
       "5": ["###", "#..", "###", "..#", "###"], "0": ["###", "#.#", "#.#", "#.#", "###"]}


def wall_blocks(text="1950", cell=.62):
    B = []
    cols = len(text) * 4 - 1
    for i, ch in enumerate(text):
        for r, row in enumerate(DIG[ch]):
            for cc, v in enumerate(row):
                if v == "#":
                    x = (i * 4 + cc - cols / 2) * cell
                    z = (4 - r) * cell
                    B.append((x, 0, z, x + cell * .96, cell * .96, z + cell * .96, (245, 245, 240)))
    return B


# ---------------------------------------------------------------- 2D helpers
def T(d, text, cx, cy, sz, fill=WHT, stroke=8, sfill=(15, 15, 20), anchor="mm"):
    d.text((cx * SS, cy * SS), text, font=font(sz), fill=fill, anchor=anchor,
           stroke_width=int(stroke * SS), stroke_fill=sfill)


def T3D(d, text, cx, cy, sz, fill, ext=(25, 25, 35), depth_px=10):
    for i in range(depth_px, 0, -1):
        d.text(((cx + i * .6) * SS, (cy + i) * SS), text, font=font(sz), fill=ext, anchor="mm",
               stroke_width=int(5 * SS), stroke_fill=ext)
    d.text((cx * SS, cy * SS), text, font=font(sz), fill=fill, anchor="mm",
           stroke_width=int(5 * SS), stroke_fill=(20, 20, 28))


def pill(d, cx, cy, text, sz, bg, fg=WHT, pad=22):
    f = font(sz)
    l, t, r, b = d.textbbox((0, 0), text, font=f, anchor="lt")
    w, h = (r - l) / SS + pad * 2, (b - t) / SS + pad * 1.2
    x0, y0 = cx - w / 2, cy - h / 2
    d.rounded_rectangle([(x0 + 6) * SS, (y0 + 7) * SS, (x0 + w + 6) * SS, (y0 + h + 7) * SS], radius=int(h / 2 * SS), fill=(15, 15, 20))
    d.rounded_rectangle([x0 * SS, y0 * SS, (x0 + w) * SS, (y0 + h) * SS], radius=int(h / 2 * SS), fill=bg,
                        outline=(15, 15, 20), width=int(4 * SS))
    d.text((cx * SS, (cy + 1) * SS), text, font=f, fill=fg, anchor="mm")


def R(d, x0, y0, x1, y1, fill, outline=(15, 15, 20), w=4, r=0):
    if r: d.rounded_rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], radius=int(r * SS), fill=fill, outline=outline, width=int(w * SS))
    else: d.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=fill, outline=outline, width=int(w * SS))


def puff(d, x, y, z, t, seed=0, n=9, size=1.0):
    """cotton-wool 'poof' used for stop-motion swaps; t in [0,1]"""
    if not (0 <= t <= 1): return
    rnd = random.Random(seed)
    sx, sy = P(x, y, z)
    for i in range(n):
        a = rnd.random() * 6.283; dist = (40 + 90 * t) * size * SS * (0.6 + rnd.random() * .6)
        rr = (34 * (1 - t) + 10) * size * SS * (0.6 + rnd.random() * .6)
        px, py = sx + math.cos(a) * dist, sy + math.sin(a) * dist * .7
        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(250, 250, 250), outline=(180, 180, 190), width=int(2 * SS))


def battery(d, cx, cy, frac, label, col=(60, 230, 120)):
    R(d, cx - 120, cy - 50, cx + 110, cy + 50, (30, 30, 36), w=6, r=14)
    R(d, cx + 110, cy - 20, cx + 130, cy + 20, (30, 30, 36), w=5, r=4)
    nb = 5
    for i in range(nb):
        if (i + 1) / nb <= frac + 1e-6:
            R(d, cx - 106 + i * 43, cy - 36, cx - 106 + i * 43 + 36, cy + 36, col, outline=None, w=0, r=5)
    T(d, label, cx, cy + 90, 40, WHT, 6)


def leaf(d, cx, cy, s=1.0, col=(70, 200, 90)):
    pts = []
    for i in range(41):
        a = math.pi * i / 40
        pts.append((cx + math.cos(a) * 0 + (-70 + 140 * i / 40) * s, cy - math.sin(a) * 50 * s))
    for i in range(41):
        a = math.pi * i / 40
        pts.append((cx + (70 - 140 * i / 40) * s, cy + math.sin(a) * 50 * s))
    d.polygon([(x * SS, y * SS) for x, y in pts], fill=col, outline=(15, 40, 20), width=int(5 * SS))
    d.line([((cx - 70 * s) * SS, cy * SS), ((cx + 75 * s) * SS, cy * SS)], fill=(20, 80, 30), width=int(5 * SS))


def checker_flag(d, cx, cy, wave_t, cells=6, cs=26):
    d.line([((cx - 10) * SS, (cy - 20) * SS), ((cx - 10) * SS, (cy + cells * cs + 120) * SS)], fill=(60, 50, 40), width=int(10 * SS))
    for i in range(cells + 2):
        for j in range(cells):
            off = math.sin(wave_t * 6 + i * .8) * 8
            x0 = cx + i * cs; y0 = cy + j * cs + off
            col = (20, 20, 20) if (i + j) % 2 else (245, 245, 245)
            d.rectangle([x0 * SS, y0 * SS, (x0 + cs) * SS, (y0 + cs) * SS], fill=col)


def malaysia_flag(d, cx, cy, w=300):
    h = w / 2
    x0, y0 = cx - w / 2, cy - h / 2
    for i in range(14):
        col = (204, 0, 1) if i % 2 == 0 else (255, 255, 255)
        d.rectangle([x0 * SS, (y0 + i * h / 14) * SS, (x0 + w) * SS, (y0 + (i + 1) * h / 14) * SS], fill=col)
    cw, chh = w / 2, h * 8 / 14
    d.rectangle([x0 * SS, y0 * SS, (x0 + cw) * SS, (y0 + chh) * SS], fill=(1, 0, 102))
    mx, my, r = x0 + cw * .36, y0 + chh / 2, chh * .38
    d.ellipse([(mx - r) * SS, (my - r) * SS, (mx + r) * SS, (my + r) * SS], fill=(255, 204, 0))
    r2 = r * .82
    d.ellipse([(mx - r2 + r * .3) * SS, (my - r2) * SS, (mx + r2 + r * .3) * SS, (my + r2) * SS], fill=(1, 0, 102))
    sx, sy, R1, R2 = x0 + cw * .7, my, chh * .3, chh * .14
    pts = []
    for i in range(28):
        rr = R1 if i % 2 == 0 else R2
        a = -math.pi / 2 + i * math.pi / 14
        pts.append(((sx + rr * math.cos(a)) * SS, (sy + rr * math.sin(a)) * SS))
    d.polygon(pts, fill=(255, 204, 0))
    d.rectangle([x0 * SS, y0 * SS, (x0 + w) * SS, (y0 + h) * SS], outline=(15, 15, 20), width=int(5 * SS))


# ---------------------------------------------------------------- anim helpers
def build(B, t, t0, rate=.07, drop=4.0, dur=.25):
    """modules drop in one by one (stop-motion assembly)"""
    out = []
    for i, b in enumerate(B):
        ti = t0 + i * rate
        if t < ti: continue
        k = max(0.0, 1 - (t - ti) / dur)
        out.append((b[0], b[1], b[2] + drop * k, b[3], b[4], b[5] + drop * k, b[6]))
    return out


def ramp(t, a, b):
    return 0.0 if t <= a else 1.0 if t >= b else (t - a) / (b - a)


def ease(u): return u * u * (3 - 2 * u)


def sent_times(scene):
    return [s[0] for s in TIMING[scene]["sentences"]]


# ---------------------------------------------------------------- scenes
# (start, end, bg_top, bg_bottom, header, era_tag, draw_fn)
def sc_hook(t, d, img, ctx):
    B = track(kerb=((60, 255, 150), (20, 20, 30)), asphalt=(40, 42, 50), grass=(25, 30, 40))
    car = rot90(car_2026(aero=0))
    th = 1.0
    cy = -14 + 14 * ease(ramp(t, .1, th)) + 3.0 * ramp(t, th, 1.6)
    objs = move(car, 0, cy, 0)
    wall = wall_blocks()
    rnd = random.Random(3)
    for i, b in enumerate(wall):
        vx, vy, vz = rnd.uniform(-5, 5), rnd.uniform(3, 9), rnd.uniform(3, 8)
        dt = max(0, t - th)
        x = vx * dt; y = vy * dt; z = vz * dt - 9.8 * dt * dt / 2
        z = max(z, -b[2])
        if dt > 0 and z <= -b[2] + 1e-6: x *= .8
        objs.append((b[0] + x, b[1] + y, b[2] + z, b[3] + x, b[4] + y, b[5] + z, b[6]))
    ctx["ground"] = B; ctx["objs"] = objs
    ctx["post"] = lambda d: (T3D(d, "76 YEARS", 540, 600, 120, (60, 255, 150)) if t > 1.15 else None,
                              T3D(d, "IN 90 SECONDS", 540, 1290, 86, WHT) if t > 1.6 else None)


def sc_1950(t, d, img, ctx):
    ctx["ground"] = track(asphalt=(95, 88, 78), grass=(120, 140, 80), kerb=((170, 60, 50), (235, 225, 205)))
    car = car_1950()
    x = 0 if t < 6.6 else (t - 6.6) * 9
    objs = move(build(car, t, .1, rate=.1), x - .5, 0, 0)
    # chequered flag made of blocks on the far side
    flag = [box(-9.2, -4.6, 0, -9.0, -4.4, 4.5, (80, 60, 40))]
    for i in range(5):
        for j in range(4):
            col = (25, 25, 25) if (i + j) % 2 else (240, 240, 240)
            wv = .25 * math.sin(t * 5 + i)
            flag.append(box(-9.0 + i * .5, -4.55 + wv * .2, 4.4 - j * .5 - .5 + wv * .1, -8.5 + i * .5, -4.45, 4.4 - j * .5 + wv * .1, col))
    objs += flag
    st = sent_times(1)
    nt = 0
    if t > st[3] - 3.2:
        nt = min(5, int((t - (st[3] - 3.2)) / .3) + 1)
    for i in range(nt):
        objs += trophy(6.5, -2.8, i * 1.45, .95)
    ctx["objs"] = objs
    ctx["post"] = lambda d: (pill(d, 820, 640, "FANGIO x5", 40, (170, 120, 20)) if nt >= 5 else None,)


def sc_1960(t, d, img, ctx):
    green = (20, 110, 60)
    ctx["ground"] = track(asphalt=(75, 80, 78), grass=(70, 140, 70))
    objs = []
    if t < 2.2:
        car = [b for b in car_1950(green) if b[6] != shade(green, .85)]
        eng = box(1.2, -.5, 1.6, 3.4, .5, 2.05, SIL)
        u1, u2, u3 = ramp(t, .5, .9), ramp(t, .9, 1.5), ramp(t, 1.5, 1.8)
        ex = -5.2 * ease(u2); ez = 1.8 * ease(u1) - 2.0 * ease(u3)
        objs = car + move([eng], ex, 0, ez)
    else:
        wing = None; stripe = None; col = green
        if t > 4.6:
            col = RED; stripe = GOLD; wing = WHT
        car = car_1960(col, wing=None, stripe=stripe)
        if wing: car = car_1960(col, wing=wing, stripe=stripe)
        x = 0 if t < 6.7 else (t - 6.7) * 10
        objs = move(car, x, 0, 0)
    ctx["objs"] = objs
    def post(d):
        puff(d, 0, 0, 1, (t - 2.05) / .4, seed=1)
        puff(d, 0, 0, 1.5, (t - 4.45) / .4, seed=2)
        if 2.3 < t < 4.4: pill(d, 540, 640, "ENGINE → BACK", 44, (30, 110, 60))
        if t > 4.7: pill(d, 540, 640, "WINGS + SPONSORS", 44, (190, 30, 35))
    ctx["post"] = post
    ctx["header"] = "1959" if t < 4.5 else "1968"


def sc_1970(t, d, img, ctx):
    ctx["ground"] = track(asphalt=(80, 74, 70), grass=(150, 140, 60), kerb=((230, 120, 20), WHT))
    st = sent_times(3)
    six = t > st[2] - 3.2 + .1
    car = car_1970(six=six)
    squash = 1.0
    if st[1] - 3.2 < t < st[2] - 3.2:
        squash = .9 if int(t * 6) % 2 else 1.0
    car = scale(car, 1.0, squash)
    x = 0 if t < 6.6 else (t - 6.6) * 10
    ctx["objs"] = move(build(car, t, 0, rate=.035), x, 0, 0)
    def post(d):
        if st[1] - 3.2 < t < st[2] - 3.2:
            for i in range(3):
                bob = 14 if int(t * 6) % 2 else 0
                cx = 380 + i * 160; y = 700 + bob
                d.polygon([((cx - 40) * SS, y * SS), ((cx + 40) * SS, y * SS), (cx * SS, (y + 50) * SS)], fill=(255, 220, 60), outline=(20, 20, 20), width=int(4 * SS))
            pill(d, 540, 620, "GROUND EFFECT", 44, (220, 110, 20))
        puff(d, 2.6, 0, .4, (t - (st[2] - 3.2)) / .4, seed=3)
        if six: pill(d, 540, 620, "6 WHEELS?!", 50, (25, 60, 165))
    ctx["post"] = post


def sc_1980(t, d, img, ctx):
    ctx["ground"] = track(asphalt=(45, 40, 60), grass=(60, 30, 90), kerb=((255, 40, 160), (40, 220, 255)))
    st = sent_times(4)
    objs = []
    tvs = st[2] - 28.2 - .3
    if t < tvs:
        x = -22 + 22 * ease(ramp(t, 0, .9))
        if t > tvs - .6: x = (t - (tvs - .6)) * 30
        car = car_1980()
        if int(t * 12) % 2 == 0 and t > .5:
            car += [box(-5.8, -.25, .55, -4.3, .25, .95, (255, 140, 20)), box(-6.4, -.15, .65, -5.8, .15, .85, (255, 230, 60))]
        objs = move(car, x, 0, 0)
    else:
        u = ease(ramp(t, tvs, tvs + .7))
        bump = .3 if int(t * 4) % 2 else 0
        objs = move(helmet_block((250, 215, 0), (30, 140, 60)), -9 + 5.2 * u + bump, 0, 0) + \
               move(helmet_block(WHT, RED), 9 - 5.2 * u - bump, 0, 0)
    ctx["objs"] = objs
    def post(d):
        if st[1] - 28.2 - .2 < t < tvs:
            hp = int(min(1, ramp(t, st[1] - 28.2 - .2, st[1] - 28.2 + 1.6)) * 1000)
            T3D(d, f"{hp}+ HP" if hp >= 1000 else f"{hp} HP", 540, 660, 100, (255, 60, 170))
        if t >= tvs + .4:
            T3D(d, "VS", 540, 820, 120, (255, 230, 60))
            T(d, "SENNA", 330, 1250, 48, (250, 215, 0)); T(d, "PROST", 760, 1250, 48, WHT)
    ctx["post"] = post


def sc_1994(t, d, img, ctx):
    ctx["ground"] = [box(-12, -6, -.5, 12, 6, 0, (60, 66, 70))]
    objs = move(build(helmet_block((235, 235, 235), (40, 120, 200)), t, .2, rate=.25, dur=.4), 0, 0, 1.0)
    objs += [box(-1.6, -1.6, 0, 1.6, 1.6, 1.0, (90, 95, 100))]
    for i in range(6):
        if t > 1.6 + i * .25:
            a = i / 6 * 6.283
            x, y = 5.2 * math.cos(a), 3.6 * math.sin(a)
            for k in range(3):
                objs.append(box(x - .5, y - .5, k * .42, x + .5, y + .5, k * .42 + .4, (30, 30, 34)))
    ctx["objs"] = objs
    def post(d):
        if t > 1.0: T3D(d, "SAFETY FIRST", 540, 640, 78, WHT)
    ctx["post"] = post


def sc_1999(t, d, img, ctx):
    ctx["ground"] = track(asphalt=(70, 70, 80), grass=(60, 150, 70), kerb=((200, 20, 30), (255, 210, 0)))
    objs = []
    # twin towers of stacked tapering blocks, built bottom-up
    for tx in (-5.0, -1.8):
        n_lv = int(min(9, ramp(t, 0, 1.4) * 9 + .5))
        for lv in range(n_lv):
            w = 1.1 - lv * .07
            objs.append(box(tx - w, -9 - w, lv * .62, tx + w, -9 + w, lv * .62 + .58, (205, 210, 220)))
        if n_lv >= 9: objs.append(box(tx - .08, -9.08, 5.58, tx + .08, -8.92, 6.6, SIL))
    if n_lv >= 6: objs.append(box(-4.4, -9.3, 2.5, -2.4, -8.7, 2.8, (170, 175, 185)))  # skybridge
    x = -26 + (t - 1.2) * 17 if t > 1.2 else -40
    objs += move(car_modern((40, 40, 46), SIL, helmet=(250, 210, 0)), x, 0, 0)
    ctx["objs"] = objs
    def post(d):
        malaysia_flag(d, 860, 700, 220)
        if t > .3: T3D(d, "SEPANG", 540, 1285, 96, (255, 210, 0))
    ctx["post"] = post


def sc_2000(t, d, img, ctx):
    ctx["ground"] = track(asphalt=(70, 66, 66), grass=(150, 40, 40), kerb=(RED, WHT))
    st = sent_times(7)
    car = car_modern(RED, (30, 30, 30), helmet=RED)
    x = 0 if t < 7.0 else (t - 7.0) * 12
    objs = move(build(car, t, 0, rate=.03), x - 1, 0, 0)
    t5 = st[1] - 46.2
    n = 0 if t < t5 else min(5, int((t - t5) / .45) + 1)
    for i in range(n):
        objs += trophy(-8 + i * 2.6, -4.6, 0, 1.1)
    t7 = st[2] - 46.2
    m = 0 if t < t7 else min(2, int((t - t7) / .35) + 1)
    for i in range(m):
        objs += trophy(6.0, -4.6, i * 1.6, 1.1)
    ctx["objs"] = objs
    def post(d):
        for i in range(n):
            sx, sy = P(-8 + i * 2.6, -4.6, 1.8)
            T(d, str(2000 + i), sx / SS, sy / SS - 30, 30, GOLD, 5)
        if m >= 2: pill(d, 540, 1290, "7 WORLD TITLES", 46, (190, 25, 30))
    ctx["post"] = post


def sc_2014(t, d, img, ctx):
    ctx["ground"] = track(asphalt=(70, 75, 82), grass=(60, 140, 140), kerb=((0, 200, 190), WHT))
    st = sent_times(8)
    th = st[2] - 54.2 + .3
    halo = t > th + .5
    car = car_modern(SIL, (0, 190, 175), helmet=(240, 240, 240), nose="low", halo=halo)
    x = -24 + 24 * ease(ramp(t, 0, 1.0))
    objs = move(car, x, 0, 0)
    if th < t <= th + .5:
        hz = 3 * (1 - ramp(t, th, th + .5))
        objs += move([box(-1.1, -.45, 1.55, .2, -.35, 1.65, DK), box(-1.1, .35, 1.55, .2, .45, 1.65, DK),
                      box(.1, -.45, 1.55, .3, .45, 1.65, DK)], x, 0, hz)
    t7 = st[1] - 54.2
    n = 0 if t < t7 else min(7, int((t - t7) / .22) + 1)
    for i in range(n):
        objs += trophy(6.5, -3.8, i * 1.25, .85)
    ctx["objs"] = objs
    ctx["header"] = "2014" if t < th else "2018"
    def post(d):
        if 1.0 < t < t7 - .1:
            battery(d, 540, 640, min(1, ramp(t, 1.0, t7 - .6)), "V6 TURBO HYBRID")
        if t7 <= t < th: pill(d, 540, 640, "HAMILTON = 7 TITLES", 44, (0, 150, 140))
        if t > th + .3: pill(d, 540, 640, "THE HALO", 48, (30, 30, 36))
    ctx["post"] = post


def sc_2021(t, d, img, ctx):
    ctx["ground"] = track(w=4.8, asphalt=(40, 44, 58), grass=(25, 40, 70), kerb=((30, 60, 200), (240, 200, 0)))
    st = sent_times(9)
    t22 = st[1] - 64.2
    if t < t22:
        a = .9 * math.sin(int(t * 4) * 1.7)
        objs = move(car_modern((25, 35, 110), RED, helmet=(30, 30, 120)), -1 + a, -2.3, 0) + \
               move(car_modern(SIL, (0, 190, 175), helmet=(240, 240, 240), nose="low"), -1 - a, 2.3, 0)
        objs = [b for b in objs]
    else:
        objs = move(build(car_modern((25, 35, 110), RED, helmet=(250, 200, 0), nose="low", halo=True, big_wheels=True),
                          t, t22, rate=.03), 0, 0, 0)
    ctx["objs"] = objs
    ctx["header"] = "2021" if t < t22 else "2022"
    def post(d):
        if t < t22:
            checker_flag(d, 760, 560, t, cells=5, cs=28)
            pill(d, 400, 640, "LAST LAP", 52, (20, 20, 24))
        else:
            pill(d, 540, 640, "GROUND EFFECT IS BACK", 42, (25, 35, 110))
    ctx["post"] = post


def sc_2026(t, d, img, ctx):
    ctx["ground"] = track(asphalt=(28, 30, 38), grass=(14, 16, 22), kerb=((60, 255, 150), (30, 34, 44)))
    st = [s - 72.2 for s in sent_times(10)]
    aero = 0.0
    if st[4] - .2 < t < st[5]:
        aero = 1.0 if int((t - st[4]) * 3) % 2 == 0 else 0.0
    s = 1.0
    if st[1] < t < st[1] + .9:
        s = 1.12 if t < st[1] + .45 else 1.0
    car = scale(car_2026(aero=aero), s)
    objs = build(car, t, .1, rate=.04)
    nteam = 0
    cols = [RED, SIL, (255, 130, 0), (25, 35, 110), (0, 120, 90), (0, 90, 200), (240, 240, 240),
            (60, 60, 70), (250, 0, 90), (120, 200, 60), (200, 170, 90)]
    if t > st[5]:
        nteam = min(11, int((t - st[5]) / .14) + 1)
        for i in range(nteam):
            objs.append(box(-7.5 + i * 1.4, -5.0, 0, -6.5 + i * 1.4, -4.0, 1.0, cols[i]))
    ctx["objs"] = objs
    def post(d):
        if st[1] <= t < st[2]: pill(d, 540, 640, "SMALLER + LIGHTER", 46, (20, 24, 32), (60, 255, 150))
        if st[2] <= t < st[3]: battery(d, 540, 640, .6 if t > st[2] + .5 else .2, "50% ELECTRIC POWER")
        if st[3] <= t < st[4]:
            leaf(d, 540, 620, 1.0); T(d, "100% SUSTAINABLE FUEL", 540, 720, 42, WHT, 6)
        if st[4] <= t < st[5]: pill(d, 540, 640, "ACTIVE AERO" + ("  ▲" if aero else "  ▼"), 50, (20, 24, 32), (60, 255, 150))
        if nteam: pill(d, 540, 640, f"{nteam} TEAMS", 54, (20, 24, 32), (60, 255, 150))
    ctx["post"] = post


def sc_cta(t, d, img, ctx):
    ctx["ground"] = track(asphalt=(28, 30, 38), grass=(14, 16, 22), kerb=((60, 255, 150), (30, 34, 44)))
    x = -26 + t * 9
    ctx["objs"] = move(car_2026(aero=1 if int(t * 3) % 2 else 0), x, 0, 0)
    def post(d):
        T3D(d, "YOUR FAV ERA?", 540, 600, 84, (60, 255, 150))
        T(d, "comment below ↓", 540, 690, 42, WHT, 6)
        pressed = t > 2.6
        sq = 6 if 2.5 < t < 2.8 else 0
        bx0, by0, bx1, by1 = 230 + sq, 1220 + sq, 850 - sq, 1350 - sq
        R(d, bx0 + 8, by0 + 10, bx1 + 8, by1 + 10, (15, 15, 20), outline=None, w=0, r=60)
        R(d, bx0, by0, bx1, by1, (60, 60, 70) if pressed else (254, 44, 85), w=6, r=60)
        T(d, "FOLLOWING ✓" if pressed else "FOLLOW", 540, (by0 + by1) / 2, 64, WHT, 0)
        T(d, "@vliamsoon", 540, 1430, 70, (60, 255, 150), 8)
        T(d, "PART 2: F1 CARS THAT GOT BANNED", 540, 1520, 38, WHT, 6)
        # tapping finger (blocky cursor)
        if 1.6 < t < 3.4:
            fx = 700 - 120 * ease(ramp(t, 1.6, 2.5)); fy = 1460 - 140 * ease(ramp(t, 1.6, 2.5))
            R(d, fx, fy, fx + 46, fy + 110, (250, 205, 160), w=5, r=20)
            R(d, fx - 10, fy + 70, fx + 70, fy + 170, (250, 205, 160), w=5, r=24)
    ctx["post"] = post


SCENES = [
    (0, 3, (8, 10, 18), (25, 30, 50), "1950 → 2026", "", sc_hook),
    (3, 12, (226, 205, 165), (170, 140, 100), "1950", "THE BEGINNING", sc_1950),
    (12, 20, (190, 225, 190), (110, 160, 120), "1959", "ROAR OF THE 60s", sc_1960),
    (20, 28, (255, 190, 110), (210, 110, 50), "1970s", "WILD IDEAS", sc_1970),
    (28, 36, (40, 10, 70), (120, 30, 140), "1980s", "TURBO ERA", sc_1980),
    (36, 42, (60, 70, 80), (30, 34, 40), "1994", "A TURNING POINT", sc_1994),
    (42, 46, (255, 170, 90), (230, 80, 70), "1999", "MALAYSIA BOLEH!", sc_1999),
    (46, 54, (120, 15, 20), (40, 5, 8), "2000-04", "RED DYNASTY", sc_2000),
    (54, 64, (200, 235, 235), (90, 140, 150), "2014", "HYBRID ERA", sc_2014),
    (64, 72, (10, 15, 45), (35, 45, 100), "2021", "SHOWDOWN", sc_2021),
    (72, 84, (4, 6, 10), (12, 40, 30), "2026", "THE NEW ERA", sc_2026),
    (84, 90, (4, 6, 10), (12, 40, 30), "", "", sc_cta),
]
CAPS = []
for sc in TIMING:
    for a, b, txt in sc["sentences"]:
        CAPS.append((a, b + .15, txt))

_bg_cache = {}
NOISE = [np.random.default_rng(i).normal(0, 1, (H, W)).astype(np.float32) for i in range(3)]


def bg(i):
    if i in _bg_cache: return _bg_cache[i]
    top, bot = np.array(SCENES[i][2], np.float32), np.array(SCENES[i][3], np.float32)
    g = np.linspace(0, 1, H * SS)[:, None, None]
    arr = (top * (1 - g) + bot * g).repeat(W * SS, 1)
    # soft vignette-ish spotlight behind the stage
    yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)
    dist = np.sqrt(((xx - CX * SS) / (W * SS * .6)) ** 2 + ((yy - CY * SS) / (H * SS * .35)) ** 2)
    arr = arr * (1.12 - .25 * np.clip(dist, 0, 1.4))[..., None]
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    _bg_cache[i] = im
    return im


def wrap(d, text, sz, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if d.textlength(test, font=font(sz)) / SS > maxw and cur: lines.append(cur); cur = w
        else: cur = test
    lines.append(cur)
    return lines


def render(fi):
    t = fi / FPS
    si = max(i for i, s in enumerate(SCENES) if s[0] <= t)
    st, en, _, _, header, tag, fn = SCENES[si]
    lt = t - st
    rnd = random.Random(fi * 7919)
    JIT[0], JIT[1] = rnd.uniform(-2.5, 2.5), rnd.uniform(-2, 2)   # stop-motion "boil"
    img = bg(si).copy()
    d = ImageDraw.Draw(img)
    ctx = {"header": header}
    fn(lt, d, img, ctx)
    ground = ctx.get("ground", [])
    draw_boxes(d, ground, outline=True)
    objs = ctx.get("objs", [])
    draw_shadows(img, objs)
    d = ImageDraw.Draw(img)
    draw_boxes(d, objs)
    post = ctx.get("post")
    JIT[0] = JIT[1] = 0
    if post: post(d)

    # ---- header: timeline progress bar + year + era tag
    hdr = ctx["header"]
    if hdr:
        prog = min(1, max(0, (t - 3) / (84 - 3)))
        R(d, 120, 196, 960, 214, (20, 20, 26), w=3, r=9)
        R(d, 120, 196, 120 + 840 * prog, 214, (60, 255, 150) if si >= 10 else (255, 60, 70), outline=None, w=0, r=9)
        T(d, "1950", 120, 240, 26, WHT, 4, anchor="lm"); T(d, "2026", 960, 240, 26, WHT, 4, anchor="rm")
        flip = lt < 0.25 and si > 0
        yy = 380 + (14 if flip else 0)
        T3D(d, hdr, 540, yy, 150 if len(hdr) <= 5 else 120, (255, 255, 255) if si != 10 else (60, 255, 150))
        if tag: pill(d, 540, 500, tag, 40, (255, 60, 70) if si not in (10,) else (20, 24, 32), WHT if si != 10 else (60, 255, 150))

    # ---- captions
    for a, b, txt in CAPS:
        if a <= t < b:
            lines = wrap(d, txt, 58, 880)
            y0 = 1440 - (len(lines) - 1) * 36 if si != 11 else 1700
            if si == 11: continue  # CTA has its own text
            for k, ln in enumerate(lines):
                T(d, ln, 540, y0 + k * 72, 58, (255, 235, 80) if k == 0 and len(lines) == 1 else WHT, 7)
            break
    T(d, "@vliamsoon", 540, 1620, 30, (255, 255, 255), 4)

    out = img.resize((W, H), Image.LANCZOS)
    # paper/felt grain + light flicker (stop-motion feel)
    a = np.asarray(out).astype(np.float32)
    a *= 1 + NOISE[fi % 3][..., None] * 0.035
    a *= rnd.uniform(.975, 1.02)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    out.save(f"{OUT}/frames/f{fi:05d}.png", compress_level=1)
    return fi


if __name__ == "__main__":
    os.makedirs(f"{OUT}/frames", exist_ok=True)
    idx = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else list(range(FRAMES))
    with Pool(4) as p:
        for k, _ in enumerate(p.imap_unordered(render, idx, chunksize=4)):
            if k % 120 == 0: print(k, flush=True)
    print("frames done")
