"""Happy Deepavali v2 — paper-doll in kurta (appears late), flying paper peacocks,
peacock-feather fan halo, lotus. Reuses helpers from deepavali.py."""
import math, random, subprocess, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
import deepavali as D
from deepavali import (W, H, FPS, DUR, NF, S, FONT_EN, FONT_TA, clamp, lerp, ease_out_back, ease_out_cubic,
                       ease_in_cubic, ease_io, ease_out_bounce, to_np, blit, xform, radial, text_sprite, pill_sprite,
                       diya_sprite, flame_sprite, gradient, kolam_sprite, toran_sprite, star_sprite, petal_sprites,
                       firework_particles, make_vig)

# ------------------------------------------------------------ paper treatment
def paperize(img, r=14, shadow=(18, 22), grain=4, seed=1):
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    pad = r + 40
    a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)))
    rgb, al = a[..., :3], a[..., 3] / 255
    if grain:
        rgb = np.clip(rgb + np.random.default_rng(seed).normal(0, grain, al.shape)[..., None], 0, 255)
    hard = (al > .4).astype(np.uint8) * 255
    outl = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    outl = cv2.GaussianBlur(outl, (3, 3), 0).astype(np.float32) / 255
    h, w = al.shape
    out = np.zeros((h, w, 4), np.float32)
    sh = cv2.GaussianBlur(outl, (0, 0), max(3, r * .8))
    dx, dy = shadow
    out[dy:, dx:, :3] = np.array([.16, .02, .12]); out[dy:, dx:, 3] = sh[:h - dy, :w - dx] * .5
    def over(rgb_, a_):
        a_ = a_[..., None]
        out[..., :3] = out[..., :3] * (1 - a_) * out[..., 3:] / np.maximum(1e-4, out[..., 3:] * (1 - a_) + a_) + rgb_ * a_ / np.maximum(1e-4, out[..., 3:] * (1 - a_) + a_)
        out[..., 3:] = out[..., 3:] * (1 - a_) + a_
    over(np.array([1.0, .995, .985]), outl)
    over(rgb / 255, al)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA")

def leaf(d, base, ang, length, width, fill, k=1, tip=1.0):
    """pointed petal/feather from base along angle (deg, ccw from +x, screen y down)."""
    bx, by = base; a = math.radians(ang)
    ux, uy = math.cos(a), -math.sin(a); px, py = -uy, ux
    pts = []
    for t in np.linspace(0, 1, 14):
        wv = math.sin(math.pi * t) ** 0.8 * width / 2
        pts.append((bx + ux * length * t + px * wv, by + uy * length * t + py * wv))
    for t in np.linspace(1, 0, 14):
        wv = math.sin(math.pi * t) ** 0.8 * width / 2
        pts.append((bx + ux * length * t - px * wv, by + uy * length * t - py * wv))
    d.polygon([(x * k, y * k) for x, y in pts], fill=fill)

# ------------------------------------------------------------ paper doll in kurta
SKIN = None
def head_layer():
    global SKIN
    c = Image.open(f"{S}/cut.png").convert("RGBA").crop((880, 180, 1480, 870))
    a = np.asarray(c).copy()
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    neck = (yy > 560) & ((xx < 180) | (xx > 452))
    a[..., 3][neck] = 0
    SKIN = tuple(int(v) for v in a[430:470, 140:200, :3].reshape(-1, 3).mean(0))
    im = Image.fromarray(a, "RGBA")
    sc = 0.63
    return im.resize((int(im.width * sc), int(im.height * sc)), Image.LANCZOS)

DW, DH, K = 760, 1300, 2
def doll_body():
    """returns (back layer: kurta, legs, dupatta), (front layer: collar, arms, hands)"""
    k = K
    def P(*pts): return [(x * k, y * k) for x, y in pts]
    teal, teal_d, teal_l = (0, 112, 150), (0, 72, 105), (40, 160, 190)
    gold, gold_d = (240, 190, 60), (170, 110, 20)
    maroon = (150, 18, 55)
    back = Image.new("RGBA", (DW * k, DH * k)); d = ImageDraw.Draw(back)
    # legs (churidar) + juttis
    for x0 in (300, 388):
        d.rounded_rectangle(P((x0, 1040), (x0 + 72, 1238)), 18 * k, fill=(248, 240, 222))
        for j in range(5):
            y = 1175 + j * 12
            d.arc(P((x0 + 4, y - 6), (x0 + 68, y + 6)), 0, 180, fill=(215, 200, 175), width=3 * k)
    for x0, dirn in ((336, -1), (424, 1)):
        d.ellipse(P((x0 - 44, 1226), (x0 + 44, 1268)), fill=gold)
        d.polygon(P((x0 + dirn * 40, 1236), (x0 + dirn * 66, 1222), (x0 + dirn * 42, 1254)), fill=gold)
        d.ellipse(P((x0 - 8, 1238), (x0 + 8, 1252)), fill=maroon)
    # kurta torso
    torso = P((300, 432), (225, 458), (203, 510), (208, 700), (192, 1000), (178, 1062), (582, 1062), (568, 1000),
              (552, 700), (557, 510), (535, 458), (460, 432))
    tm = Image.new("L", back.size); ImageDraw.Draw(tm).polygon(torso, fill=255)
    shade = np.zeros((DH * k, DW * k, 4), np.uint8)
    xs = np.linspace(0, 1, DW * k)
    cyl = np.clip(1 - ((xs - 0.47) / 0.36) ** 2, 0, 1) ** 0.6
    for c in range(3): shade[..., c] = (teal_d[c] + (teal_l[c] - teal_d[c]) * cyl)[None, :]
    shade[..., 3] = np.asarray(tm)
    back.alpha_composite(Image.fromarray(shade, "RGBA"))
    # butti (small gold flowers) clipped to torso
    bt = Image.new("RGBA", back.size); bd = ImageDraw.Draw(bt)
    for row, y in enumerate(range(520, 990, 58)):
        for x in range(220 + (row % 2) * 29, 560, 58):
            for ang in range(0, 360, 90):
                ox, oy = 5 * math.cos(math.radians(ang)), 5 * math.sin(math.radians(ang))
                bd.ellipse(P((x + ox - 3.5, y + oy - 3.5), (x + ox + 3.5, y + oy + 3.5)), fill=gold + (230,))
    bt.putalpha(Image.fromarray(np.minimum(np.asarray(bt)[..., 3], np.asarray(tm))))
    back.alpha_composite(bt)
    # fold lines, side slits
    for (a_, b_) in [((265, 760), (250, 995)), ((498, 760), (512, 995)), ((330, 860), (325, 1000))]:
        d.line(P(a_, b_), fill=teal_d, width=3 * k)
    d.line(P((196, 960), (186, 1060)), fill=(0, 50, 70), width=4 * k)
    d.line(P((564, 960), (574, 1060)), fill=(0, 50, 70), width=4 * k)
    # hem borders
    d.polygon(P((180, 1018), (580, 1018), (582, 1062), (178, 1062)), fill=gold)
    d.line(P((186, 1004), (574, 1004)), fill=gold, width=5 * k)
    for x in range(196, 570, 26):
        d.ellipse(P((x - 6, 1034), (x + 6, 1046)), fill=maroon)
        d.polygon(P((x + 13, 1028), (x + 18, 1040), (x + 13, 1052), (x + 8, 1040)), fill=(255, 245, 200))
    # placket + buttons
    d.rectangle(P((371, 470), (389, 735)), fill=gold)
    for y in (500, 548, 596, 644, 692):
        d.ellipse(P((372, y - 7), (388, y + 7)), fill=(255, 248, 225)); d.ellipse(P((377, y - 2), (383, y + 4)), fill=gold_d)
    # dupatta (maroon stole) across the torso
    dup = P((232, 452), (318, 436), (585, 905), (548, 968))
    d.polygon(dup, fill=maroon)
    d.line(P((232, 452), (548, 968)), fill=gold, width=9 * k)
    d.line(P((318, 436), (585, 905)), fill=gold, width=9 * k)
    for t in np.linspace(0.08, 0.92, 12):
        x = lerp(275, 566, t); y = lerp(444, 936, t)
        d.ellipse(P((x - 6, y - 6), (x + 6, y + 6)), fill=(255, 200, 80))
    for j in range(5):
        x = lerp(548, 585, j / 4); y = lerp(968, 905, j / 4)
        d.line(P((x, y), (x + 6, y + 30)), fill=gold, width=4 * k); d.ellipse(P((x + 1, y + 26), (x + 11, y + 36)), fill=gold)

    front = Image.new("RGBA", back.size); f = ImageDraw.Draw(front)
    # Nehru collar (two halves)
    f.polygon(P((288, 418), (373, 424), (373, 474), (282, 470)), fill=gold)
    f.polygon(P((472, 418), (387, 424), (387, 474), (478, 470)), fill=gold)
    for x in range(296, 370, 14): f.ellipse(P((x - 3, 444), (x + 3, 450)), fill=maroon)
    for x in range(394, 470, 14): f.ellipse(P((x - 3, 444), (x + 3, 450)), fill=maroon)
    # arms in namaste
    for sgn in (1, -1):
        def X(x): return x if sgn == 1 else DW - x
        sh, el, wr = (X(238), 492), (X(244), 735), (X(352), 642)
        for col, ex in ((teal_d, 7), (teal, 0)):
            f.line(P(sh, el), fill=col, width=(86 + ex * 2) * k); f.line(P(el, wr), fill=col, width=(78 + ex * 2) * k)
            f.ellipse(P((el[0] - 40 - ex, el[1] - 40 - ex), (el[0] + 40 + ex, el[1] + 40 + ex)), fill=col)
            f.ellipse(P((sh[0] - 43 - ex, sh[1] - 43 - ex), (sh[0] + 43 + ex, sh[1] + 43 + ex)), fill=col)
        f.line(P(sh, el), fill=teal, width=86 * k); f.ellipse(P((sh[0] - 43, sh[1] - 43), (sh[0] + 43, sh[1] + 43)), fill=teal)
        f.ellipse(P((el[0] - 40, el[1] - 40), (el[0] + 40, el[1] + 40)), fill=teal)
        f.line(P(el, wr), fill=teal, width=78 * k)
        f.line(P(el, (lerp(el[0], wr[0], .5), lerp(el[1], wr[1], .5))), fill=teal_d, width=4 * k)
        cuff_a = (lerp(el[0], wr[0], .8), lerp(el[1], wr[1], .8))
        f.line(P(cuff_a, wr), fill=gold, width=80 * k)
        for pin in (sh, el):
            f.ellipse(P((pin[0] - 10, pin[1] - 10), (pin[0] + 10, pin[1] + 10)), fill=(255, 215, 90), outline=gold_d, width=2 * k)
            f.ellipse(P((pin[0] - 4, pin[1] - 6), (pin[0] + 1, pin[1] - 1)), fill=(255, 255, 230))
    skin = (238, 198, 168); skin_d = (205, 150, 120)
    f.polygon(P((380, 512), (346, 590), (346, 662), (380, 676), (414, 662), (414, 590)), fill=skin)
    f.line(P((380, 522), (380, 672)), fill=skin_d, width=3 * k)
    for x in (364, 396): f.ellipse(P((x - 9, 600), (x + 9, 640)), fill=skin_d)
    for x in (352, 408):  # bangles / kada
        f.line(P((x - 8, 650), (x + 8, 662)), fill=(255, 220, 110), width=7 * k)
    return back.resize((DW, DH), Image.LANCZOS), front.resize((DW, DH), Image.LANCZOS)

NECK = (380, 444)
def doll_frame(head_ang, head_dy):
    A = ASSETS
    im = A["doll_back"].copy()
    hd = A["head"].rotate(head_ang, resample=Image.BICUBIC, expand=False, center=(A["head"].width / 2, A["head"].height - 40))
    hx = int(NECK[0] - hd.width / 2); hy = int(NECK[1] + 30 - hd.height + head_dy)
    lay = Image.new("RGBA", im.size); lay.alpha_composite(hd, (hx, max(0, hy)))
    la = np.asarray(lay).copy(); la[470:, :, 3] = 0
    im.alpha_composite(Image.fromarray(la, "RGBA"))
    im.alpha_composite(A["doll_front"])
    return paperize(im, r=13, grain=3)

# ------------------------------------------------------------ peacocks
def eye_ellipses(d, cx, cy, s, k):
    for (w_, h_, col) in [(34, 28, (205, 165, 55)), (27, 22, (60, 160, 90)), (20, 16, (20, 150, 160)), (13, 11, (25, 70, 200)), (7, 6, (12, 18, 70))]:
        d.ellipse(((cx - w_ * s / 2) * k, (cy - h_ * s / 2) * k, (cx + w_ * s / 2) * k, (cy + h_ * s / 2) * k), fill=col)

def flying_peacock(phase):
    k = 2; w, h = 600, 360
    im = Image.new("RGBA", (w * k, h * k)); d = ImageDraw.Draw(im)
    O = (262, 182)
    # train
    for i in range(9):
        ang = 168 + i * 3.6 + 5 * math.sin(phase + i * .6)
        L = 245 - abs(i - 4) * 12
        a = math.radians(ang); tip = (O[0] + math.cos(a) * L, O[1] - math.sin(a) * L)
        leaf(d, O, ang, L, 30, (40, 125, 70) if i % 2 else (70, 150, 60), k)
        d.line(((O[0]) * k, O[1] * k, tip[0] * k, tip[1] * k), fill=(200, 210, 150), width=2 * k)
        eye_ellipses(d, tip[0] + math.cos(a) * -14, tip[1] - math.sin(a) * -14, 1.0, k)
    # far wing
    s = math.sin(phase)
    def wing(col1, col2, off, dark=1.0):
        base = (300 + off, 150)
        th = 160 - 75 * s
        for j in range(5):
            leaf(d, base, th - 30 + j * 13, 130 - j * 6, 34, tuple(int(c * dark) for c in col1), k)
        leaf(d, base, th - 8, 72, 60, tuple(int(c * dark) for c in col2), k)
        for j in range(4):
            a = math.radians(th - 20 + j * 12); r = 40
            x, y = base[0] + math.cos(a) * r, base[1] - math.sin(a) * r
            d.ellipse(((x - 4) * k, (y - 4) * k, (x + 4) * k, (y + 4) * k), fill=(240, 230, 200))
    wing((150, 85, 40), (30, 90, 120), 18, .7)
    # body
    d.ellipse(((300 - 72) * k, (182 - 44) * k, (300 + 72) * k, (182 + 44) * k), fill=(25, 75, 210))
    d.ellipse(((300 - 40) * k, (182 - 30) * k, (300 + 38) * k, (182 + 4) * k), fill=(60, 120, 240))
    d.polygon([(x * k, y * k) for x, y in [(335, 150), (362, 160), (392, 98), (372, 86)]], fill=(30, 100, 230))
    d.ellipse(((384 - 24) * k, (82 - 24) * k, (384 + 24) * k, (82 + 24) * k), fill=(30, 100, 230))
    d.ellipse(((386 - 15) * k, (78 - 5) * k, (386 + 15) * k, (78 + 5) * k), fill=(255, 255, 255))
    d.ellipse(((390 - 5) * k, (80 - 5) * k, (390 + 5) * k, (80 + 5) * k), fill=(10, 10, 20))
    d.ellipse(((391) * k, (77) * k, (393) * k, (79) * k), fill=(255, 255, 255))
    d.ellipse(((396 - 7) * k, (90 - 4) * k, (396 + 7) * k, (90 + 4) * k), fill=(255, 140, 170))
    d.polygon([(x * k, y * k) for x, y in [(404, 76), (428, 86), (404, 94)]], fill=(235, 205, 150))
    for j, (ex, ey) in enumerate([(362, 34), (372, 26), (384, 24), (396, 28)]):
        d.line((380 * k, 62 * k, ex * k, ey * k), fill=(40, 90, 200), width=2 * k)
        d.ellipse(((ex - 5) * k, (ey - 5) * k, (ex + 5) * k, (ey + 5) * k), fill=(40, 170, 140) if j % 2 else (30, 90, 220))
    wing((190, 110, 55), (40, 120, 150), 0)
    im = im.resize((w, h), Image.LANCZOS)
    return paperize(im, r=7, shadow=(10, 12), grain=0)

def feather_sprite(L, wmax=104):
    k = 2; w, h = wmax + 20, L
    im = Image.new("RGBA", (w * k, h * k)); d = ImageDraw.Draw(im)
    cx = w / 2
    rng = random.Random(L)
    for j in range(120):  # fluffy barbs
        t = j / 119
        y = lerp(h * .62, h * .06, t)
        span = math.sin(math.pi * t) ** .7 * wmax / 2
        for sgn in (-1, 1):
            col = (int(lerp(60, 150, t)), int(lerp(140, 175, t)), int(lerp(70, 60, t)))
            d.line((cx * k, y * k, (cx + sgn * span) * k, (y - 18) * k), fill=col + (220,), width=2 * k)
    d.line((cx * k, h * k, cx * k, h * .05 * k), fill=(235, 225, 170), width=3 * k)
    cy = h * .14
    for (w_, h_, col) in [(84, 104, (190, 150, 55)), (70, 88, (120, 170, 60)), (54, 68, (20, 165, 150)), (38, 48, (25, 70, 200)), (22, 28, (12, 18, 70))]:
        d.ellipse(((cx - w_ / 2) * k, (cy - h_ / 2 + (104 - h_) * .18) * k, (cx + w_ / 2) * k, (cy + h_ / 2 + (104 - h_) * .18) * k), fill=col)
    return im.resize((w, h), Image.LANCZOS)

def fan_sprite(p, outer, inner, lotus):
    w, h = 1360, 720; C = (680, 690)
    im = Image.new("RGBA", (w, h))
    def place(spr, theta):
        r = spr.rotate(theta - 90, resample=Image.BICUBIC, expand=True)
        # base of feather (bottom centre) must sit on C: rotate base vector
        L = spr.height / 2; a = math.radians(theta)
        cx, cy = C[0] + math.cos(a) * L, C[1] - math.sin(a) * L
        im.alpha_composite(r, (int(cx - r.width / 2), int(cy - r.height / 2)))
    n = 25
    for i in range(n):
        th = 8 + 164 * i / (n - 1); place(outer, 90 + (th - 90) * p)
    for i in range(14):
        th = 14 + 152 * (i + .5) / 14; place(inner, 90 + (th - 90) * p)
    im.alpha_composite(lotus, (int(C[0] - lotus.width / 2), int(C[1] - lotus.height * .75)))
    return paperize(im, r=6, shadow=(10, 12), grain=0)

def lotus_sprite(size=260):
    k = 2; s = size
    im = Image.new("RGBA", (s * k, s * k)); d = ImageDraw.Draw(im)
    base = (s / 2, s * .78)
    for ang, L, col in [(165, .40, (235, 90, 150)), (15, .40, (235, 90, 150)), (140, .48, (245, 120, 170)), (40, .48, (245, 120, 170)),
                        (115, .55, (255, 150, 190)), (65, .55, (255, 150, 190)), (90, .6, (255, 185, 215))]:
        leaf(d, base, ang, s * L, s * .26, col, k)
        a = math.radians(ang)
        d.line((base[0] * k, base[1] * k, (base[0] + math.cos(a) * s * L * .8) * k, (base[1] - math.sin(a) * s * L * .8) * k), fill=(255, 220, 235), width=2 * k)
    d.ellipse(((s / 2 - 22) * k, (s * .78 - 14) * k, (s / 2 + 22) * k, (s * .78 + 10) * k), fill=(255, 200, 60))
    d.ellipse(((s * .1) * k, (s * .8) * k, (s * .9) * k, (s * .95) * k), fill=(40, 140, 80))
    return im.resize((s, s), Image.LANCZOS)

# ------------------------------------------------------------ timeline
TEXT = [
    ("one",   470, 0.25, 1.72, "pop"),
    ("tiny",  620, 0.65, 1.72, "pop"),
    ("light", 790, 1.05, 1.72, "pop"),
    ("can",   480, 2.00, 3.00, "pop"),
    ("every", 640, 2.50, 3.00, "slam"),
    ("happy", 560, 5.00, 9.00, "drop"),
    ("deep",  740, 5.50, 9.00, "slam"),
    ("ta1",   930, 6.50, 9.00, "pop"),
    ("ta2",  1068, 7.00, 9.00, "pop"),
    ("translit", 1170, 7.50, 9.00, "fade"),
    ("w1",    230, 9.50, 12.20, "pop"),
    ("w2",    330, 10.00, 12.20, "pop"),
    ("w3",    462, 10.50, 12.20, "slam"),
    ("w4",    600, 11.00, 12.20, "slam"),
    ("o1",    250, 12.50, 99, "drop"),
    ("o2",    390, 13.00, 99, "pop"),
    ("o3",    500, 13.50, 99, "fade"),
    ("sign",  600, 13.75, 99, "pop"),
]
SLAMS = [t for k_, y, t, o, s in TEXT if s == "slam"]
FIREWORKS = [(8.0, 230, 330, 0), (8.5, 850, 300, 1), (11.6, 900, 160, 2), (13.0, 540, 170, 0), (13.25, 880, 330, 2), (13.4, 210, 360, 1)]
DIYA_X = [140, 340, 540, 740, 940]
DIYA_LIT = [3.75, 4.0, 4.25, 4.5]
DIYA_ORDER = [0, 1, 3, 4]
FAN_OPEN = 4.8
DOLL_IN = 9.0
DOLL_HOP = [(10.5, -55), (11.5, 55), (12.5, 0)]
FAN_C = (540, 1190)
# peacock flights: (t0, t1, [(x,y) control points], scale, behind_doll)
FLIGHTS = [
    (3.0, 5.3, [(-330, 1180), (300, 1000), (700, 760), (1420, 560)], 1.0, False),
    (5.2, 8.6, [(1400, 380), (900, 260), (300, 330), (-350, 250)], 0.62, False),
    (6.4, 9.2, [(-330, 1480), (350, 1350), (800, 1420), (1420, 1300)], 0.78, False),
    (9.2, 12.0, [(1420, 900), (800, 760), (250, 880), (-350, 700)], 0.7, True),
    (12.3, 15.2, [(-330, 1500), (250, 1200), (700, 960), (1450, 760)], 0.66, False),
]

def bez(pts, u):
    p = list(pts)
    while len(p) > 1: p = [(lerp(a[0], b[0], u), lerp(a[1], b[1], u)) for a, b in zip(p, p[1:])]
    return p[0]

def build_assets():
    A = {}
    A["bg_dark"] = gradient([(0, (8, 6, 30)), (.6, (25, 10, 55)), (1, (45, 12, 60))])
    A["bg_warm"] = gradient([(0, (70, 8, 70)), (.35, (160, 22, 85)), (.7, (230, 80, 45)), (1, (255, 155, 50))])
    A["kolam"] = kolam_sprite(); A["toran"] = to_np(toran_sprite())
    for nm, lit, sc in [("diya_off", False, 1), ("diya_on", True, 1), ("diya_off_s", False, .62), ("diya_on_s", True, .62)]:
        A[nm] = diya_sprite(lit, sc)
    A["flames"] = [flame_sprite(i, 1.0) for i in range(8)]; A["flames_s"] = [flame_sprite(i, .62) for i in range(8)]
    A["glow"] = radial(520, (255, 170, 60), 2.2, .9); A["glow_s"] = radial(300, (255, 170, 60), 2.2, .8)
    A["halo"] = radial(1300, (255, 200, 120), 1.6, .55)
    A["bokeh"] = [radial(s, c, 1.3, .5) for s, c in [(60, (255, 210, 120)), (90, (255, 150, 90)), (40, (255, 240, 200)), (120, (80, 220, 200))]]
    A["star"] = star_sprite(90); A["star_s"] = star_sprite(44)
    A["dot"] = radial(38, (255, 230, 160), 1.2, 1.0); A["dot2"] = radial(38, (255, 120, 200), 1.2, 1.0); A["dot3"] = radial(38, (140, 255, 200), 1.2, 1.0)
    A["gold"] = radial(22, (255, 220, 120), 1.2, 1.0)
    A["petals"] = petal_sprites()
    A["head"] = head_layer()
    A["doll_back"], A["doll_front"] = doll_body()
    A["peacock_r"] = [flying_peacock(2 * math.pi * i / 8) for i in range(8)]
    A["peacock_l"] = [p.transpose(Image.FLIP_LEFT_RIGHT) for p in A["peacock_r"]]
    lot = lotus_sprite(230)
    outer, inner = feather_sprite(600), feather_sprite(440, 90)
    A["fan"] = [fan_sprite(p, outer, inner, lot) for p in np.linspace(0.05, 1, 10)]
    A["lotus_c"] = paperize(lotus_sprite(210), r=6, shadow=(8, 10), grain=0)
    gold = dict(top=(255, 245, 170), bot=(255, 160, 20))
    pink = dict(top=(255, 225, 245), bot=(255, 110, 190), ext=(120, 20, 90))
    white = dict(top=(255, 255, 255), bot=(255, 236, 210), ext=(70, 5, 45), outline=(110, 15, 70))
    T = {}
    T["one"] = text_sprite("One", FONT_EN, 120, **white, depth=8, stroke=0)
    T["tiny"] = text_sprite("tiny", FONT_EN, 120, **white, depth=8, stroke=0)
    T["light"] = text_sprite("light...", FONT_EN, 140, **gold, depth=10, stroke=0)
    T["can"] = text_sprite("can light up", FONT_EN, 100, **white, stroke=10)
    T["every"] = text_sprite("EVERYTHING!", FONT_EN, 150, **gold, stroke=14, depth=14)
    T["happy"] = text_sprite("HAPPY", FONT_EN, 150, **pink, stroke=14, depth=14)
    T["deep"] = text_sprite("DEEPAVALI", FONT_EN, 170, **gold, stroke=16, depth=16)
    T["ta1"] = text_sprite("தீபாவளி", FONT_TA, 120, var="ExtraBold", **white, stroke=12, depth=10)
    T["ta2"] = text_sprite("நல்வாழ்த்துகள்!", FONT_TA, 104, var="ExtraBold", **gold, stroke=12, depth=10)
    T["translit"] = text_sprite("Deepavali Nalvazhthukal", FONT_EN, 48, var="SemiBold", **white, stroke=6, depth=4)
    T["w1"] = text_sprite("May the light of Deepavali", FONT_EN, 70, var="SemiBold", **white, stroke=8, depth=6)
    T["w2"] = text_sprite("fill your home with", FONT_EN, 70, var="SemiBold", **white, stroke=8, depth=6)
    T["w3"] = text_sprite("JOY, PEACE", FONT_EN, 130, **gold, stroke=14, depth=14)
    T["w4"] = text_sprite("& PROSPERITY", FONT_EN, 112, **pink, stroke=14, depth=12)
    T["o1"] = text_sprite("Happy Deepavali!", FONT_EN, 112, **gold, stroke=12, depth=12)
    T["o2"] = text_sprite("தீபாவளி நல்வாழ்த்துகள்!", FONT_TA, 74, var="ExtraBold", **white, stroke=10, depth=8)
    T["o3"] = text_sprite("Warm wishes from", FONT_EN, 52, var="Medium", **white, stroke=6, depth=4)
    T["sign"] = pill_sprite("W/LL/A///.  |  williamsoon.com", 46)
    A["T"] = T
    return A

def doll_state(t):
    tq = math.floor(t * 12) / 12
    if tq < DOLL_IN: return None
    rng = random.Random(int(tq * 12) * 13 + 5)
    u = tq - DOLL_IN
    y = 1300 * (1 - ease_out_back(u / 0.55, 1.8))
    x, px, hop, tilt, sq = 0, 0, 0, 0, 1.0
    for th, nx in DOLL_HOP:
        if tq < th: break
        v = (tq - th) / 0.42
        if v < 1:
            x = lerp(px, nx, ease_io(v)); hop = -95 * math.sin(math.pi * v); tilt = (nx - px) * .07 * math.sin(math.pi * v)
        else:
            x = nx
            if tq - th - .42 < .2: sq = 1 - .05 * math.sin(math.pi * (tq - th - .42) / .2)
        px = nx
    nod = 16 * math.sin(math.pi * clamp((tq - 9.7) / 0.5))  # namaste bow
    head_ang = 4 * math.sin(tq * 2.6) + rng.uniform(-1.2, 1.2)
    return dict(x=x, y=y + hop + rng.uniform(-3, 3), ang=tilt + rng.uniform(-1, 1) + 1.5 * math.sin(tq * 2), sq=sq,
                head_ang=head_ang, head_dy=nod)

def draw_flight(frame, t, fl, A):
    t0, t1, pts, sc, _ = fl
    if not (t0 <= t <= t1): return
    u = (t - t0) / (t1 - t0)
    x, y = bez(pts, ease_io(u) * .8 + u * .2)
    y += 25 * math.sin(t * 5)
    x2, y2 = bez(pts, min(1, ease_io(u) * .8 + u * .2 + .01))
    right = x2 >= x
    # gold sparkle trail
    for j in range(1, 16):
        tb = t - j * 0.045
        if tb < t0: break
        ub = (tb - t0) / (t1 - t0); bx, by = bez(pts, ease_io(ub) * .8 + ub * .2); by += 25 * math.sin(tb * 5)
        r = random.Random(j * 31 + int(tb * 30))
        blit(frame, A["gold"], bx + r.uniform(-30, 30) - (110 if right else -110) * sc, by + r.uniform(-25, 25) + 20 * sc, (1 - j / 16) * .9, add=True)
    stop = math.floor(t * 12)
    spr = (A["peacock_r"] if right else A["peacock_l"])[stop % 8]
    ang = math.degrees(math.atan2(-(y2 - y), (x2 - x))) if right else -math.degrees(math.atan2(-(y2 - y), (x - x2)))
    ang = max(-18, min(18, ang * .6))
    blit(frame, xform(spr, sc, ang), x, y, 1.0)

def render_frame(fi):
    t = fi / FPS; A = ASSETS
    reveal = ease_out_cubic((t - 2.0) / 0.6)
    if t < 2.0: frame = A["bg_dark"].copy()
    elif reveal >= 1: frame = A["bg_warm"].copy()
    else:
        yy, xx = np.ogrid[0:H, 0:W]
        m = np.clip((reveal * 2300 - np.sqrt((xx - 540) ** 2 + (yy - 1250) ** 2)) / 60, 0, 1)[..., None].astype(np.float32)
        frame = A["bg_dark"] * (1 - m) + A["bg_warm"] * m
    if t >= 2.0:
        blit(frame, to_np(A["kolam"].rotate(-t * 8, resample=Image.BILINEAR)), 540, 1080, .16 * reveal)
    rng = random.Random(5)
    for i in range(26):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H); vs = rng.uniform(20, 60); ph = rng.uniform(0, 6)
        blit(frame, A["bokeh"][i % 4], x0 + 30 * math.sin(t * .8 + ph), (y0 - vs * t) % (H + 200) - 100,
             (.25 if t < 2 else .5) * (.6 + .4 * math.sin(t * 2 + ph)), add=True)
    if t >= 2.2:
        blit(frame, A["toran"], 540 + 4 * math.sin(t * 1.7), lerp(-120, 100, ease_out_bounce((t - 2.2) / .6)), 1.0)
    # ---- sacred peacock-feather fan + halo
    if t >= FAN_OPEN:
        p = clamp((t - FAN_OPEN) / 0.6)
        idx = min(9, int(ease_out_back(p, 1.2) * 9.99)) if p < 1 else 9
        blit(frame, A["halo"], FAN_C[0], FAN_C[1] - 250, (.5 + .1 * math.sin(t * 3)) * p, add=True)
        fan = A["fan"][max(0, idx)]
        sc = lerp(.85, 1.0, ease_out_back(p)) if p < 1 else 1 + .012 * math.sin(t * 2.4)
        spr = xform(fan, sc, 1.5 * math.sin(t * 1.3))
        blit(frame, spr, FAN_C[0], FAN_C[1] - fan.height * sc / 2 + 40, min(1, p * 2) * .95)
    for fl in FLIGHTS:
        if fl[4]: draw_flight(frame, t, fl, A)
    # ---- paper doll
    ds = doll_state(t)
    if ds:
        key = (round(ds["head_ang"], 1), round(ds["head_dy"]))
        dimg = doll_frame(ds["head_ang"], ds["head_dy"])
        sc = .9
        spr = xform(dimg, sc, ds["ang"])
        if ds["sq"] != 1:
            spr = to_np(Image.fromarray((spr * 255).astype(np.uint8)).resize((int(spr.shape[1] * (2 - ds["sq"])), int(spr.shape[0] * ds["sq"]))))
        blit(frame, spr, 540 + ds["x"], 712 + spr.shape[0] / 2 + ds["y"], 1.0)
        if 0 <= t - DOLL_IN < .5:  # entry sparkle burst
            r = random.Random(fi)
            for j in range(14):
                blit(frame, A["star_s"], 540 + r.uniform(-330, 330), 1000 + r.uniform(-300, 300), 1 - (t - DOLL_IN) / .5, add=True)
    # ---- lamps + lotus
    stop = math.floor(t * 12) / 12
    if t < 3.0:
        dx, dy = 540, 1250; lit = t >= 1.75
        blit(frame, to_np(A["diya_on"] if lit else A["diya_off"]), dx, dy, clamp(t / .3))
        if lit:
            g = clamp((t - 1.75) / .25)
            blit(frame, A["glow"], dx + 80, dy - 90, g * (.9 + .1 * math.sin(t * 20)), add=True)
            blit(frame, A["flames"][int(stop * 12) % 8], dx + 92, dy - 115 - 10 * (1 - g), g)
        if 1.5 <= t < 1.8:
            for j in range(10):
                r = random.Random(fi * 10 + j)
                blit(frame, A["dot"], dx + 92 + r.uniform(-40, 40), dy - 100 + r.uniform(-40, 30), 1.0, add=True)
    else:
        for side, lx in ((-1, 52), (1, W - 52)):
            e = ease_out_back((t - 3.1) / .5)
            if e > 0: blit(frame, to_np(A["lotus_c"]), lx, lerp(2100, 1800, e) + 5 * math.sin(t * 2 + side), 1.0)
        for i, x in enumerate(DIYA_X):
            enter = ease_out_back((t - 3.0 - i * .06) / .4)
            if enter <= 0: continue
            y = lerp(2100, 1790, enter) + 6 * math.sin(stop * 6 + i)
            lit_t = 1.75 if i == 2 else DIYA_LIT[DIYA_ORDER.index(i)]
            lit = t >= lit_t; sq = 1.0
            if lit and t - lit_t < .25: sq = 1 + .15 * math.sin(math.pi * (t - lit_t) / .25)
            d_ = A["diya_on_s"] if lit else A["diya_off_s"]
            spr = to_np(d_) if sq == 1 else to_np(d_.resize((int(d_.width * (2 - sq)), int(d_.height * sq))))
            blit(frame, spr, x, y - (sq - 1) * 40, 1.0)
            if lit:
                blit(frame, A["glow_s"], x + 50, y - 60, .85 + .15 * math.sin(t * 17 + i), add=True)
                blit(frame, A["flames_s"][(int(stop * 12) + i * 3) % 8], x + 57, y - 72, 1.0)
    if t >= 5.5:
        r = random.Random(11)
        for i in range(30):
            x0 = r.uniform(0, W); st = 5.5 + r.uniform(0, 3); vs = r.uniform(180, 320); ph = r.uniform(0, 6)
            if t < st: continue
            y = -40 + (t - st) * vs
            if y > H + 40: continue
            blit(frame, A["petals"][int(t * 6 + i) % 12], x0 + 50 * math.sin((t - st) * 2 + ph), y, .95)
    for fl in FLIGHTS:
        if not fl[4]: draw_flight(frame, t, fl, A)
    for (ft, fx, fy, kind) in FIREWORKS:
        pts, rocket = firework_particles(t, ft, fx, fy, kind)
        dot = [A["dot"], A["dot2"], A["dot3"]][kind]
        if rocket: blit(frame, A["dot"], rocket[0], rocket[1], 1.0, add=True)
        for (x, y, op) in pts: blit(frame, dot, x, y, op * 1.4, add=True)
        if 0 <= t - ft < .12: blit(frame, A["glow"], fx, fy, 1 - (t - ft) / .12, add=True)
    T = A["T"]
    for key, y, t0, t1, style in TEXT:
        st = D.text_anim(t, t0, t1, style)
        if not st: continue
        sc, ang, dy, op = st
        spr = xform(T[key], sc, ang)
        if spr is not None: blit(frame, spr, 540, y + dy, op)
    if t >= 2.0:
        r = random.Random(21)
        for i in range(14):
            x = r.uniform(60, W - 60); y = r.uniform(140, 1200); ph = r.uniform(0, 6); sp = r.uniform(3, 6)
            s = max(0, math.sin(t * sp + ph)) ** 3
            if s > .05: blit(frame, A["star_s"] if i % 2 else A["star"], x, y, s, add=True)
    for ft, strength in [(2.0, .85), (5.5, .3), (DOLL_IN, .25), (13.0, .2)]:
        if 0 <= t - ft < .25: frame += strength * (1 - (t - ft) / .25)
    frame *= VIG
    for ts in SLAMS + [2.0]:
        if 0 <= t - ts < .2:
            a = 14 * (1 - (t - ts) / .2)
            frame = np.roll(frame, (int(a * math.sin(fi * 2.1)), int(a * math.cos(fi * 1.7))), axis=(0, 1))
    if t > DUR - .4: frame *= clamp((DUR - t) / .4) * .5 + .5
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8).tobytes()

ASSETS = None; VIG = None
def init():
    global ASSETS, VIG
    ASSETS = build_assets(); VIG = make_vig()

if __name__ == "__main__":
    init()
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for tt in [float(x) for x in sys.argv[2:]]:
            Image.frombytes("RGB", (W, H), render_frame(int(tt * FPS))).save(f"{S}/v2_{tt:05.2f}.jpg", quality=88)
        sys.exit()
    from multiprocessing import Pool
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", f"{S}/video2_noaudio.mp4"], stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(NF), chunksize=4)):
            ff.stdin.write(b)
            if i % 90 == 0: print("frame", i, flush=True)
    ff.stdin.close(); ff.wait(); print("done")
