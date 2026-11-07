import math, random, subprocess, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

S = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, DUR = 1080, 1920, 30, 15.0
NF = int(FPS * DUR)
FONT_EN = f"{S}/fonts/Fredoka.ttf"
FONT_TA = f"{S}/fonts/BalooThambi2.ttf"

# ----------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lerp(a, b, t): return a + (b - a) * t
def ease_out_back(t, s=1.9):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1
def ease_out_cubic(t): t = clamp(t); return 1 - (1 - t) ** 3
def ease_in_cubic(t): t = clamp(t); return t ** 3
def ease_io(t): t = clamp(t); return t * t * (3 - 2 * t)
def ease_out_bounce(t):
    t = clamp(t); n, d = 7.5625, 2.75
    if t < 1 / d: return n * t * t
    if t < 2 / d: t -= 1.5 / d; return n * t * t + .75
    if t < 2.5 / d: t -= 2.25 / d; return n * t * t + .9375
    t -= 2.625 / d; return n * t * t + .984375

def to_np(img): return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0

def blit(dst, spr, cx, cy, op=1.0, add=False):
    """spr: float HxWx4 straight alpha. cx,cy = centre."""
    if op <= 0.001 or spr is None: return
    h, w = spr.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0: return
    s = spr[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = s[..., 3:4] * op
    d = dst[Y0:Y1, X0:X1]
    if add: d += s[..., :3] * a
    else: d *= (1 - a); d += s[..., :3] * a

def xform(pil, scale=1.0, angle=0.0):
    if scale <= 0.01: return None
    im = pil
    if abs(scale - 1) > 1e-3:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BICUBIC)
    if abs(angle) > 0.05:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    return to_np(im)

def radial(size, color, power=2.0, inner=1.0):
    r = size / 2
    y, x = np.mgrid[0:size, 0:size]
    d = np.sqrt((x - r + .5) ** 2 + (y - r + .5) ** 2) / r
    a = np.clip(1 - d, 0, 1) ** power * inner
    out = np.zeros((size, size, 4), np.float32)
    out[..., :3] = np.array(color, np.float32) / 255
    out[..., 3] = a
    return out

# ----------------------------------------------------------------- text sprites
def font(path, size, var):
    f = ImageFont.truetype(path, size)
    f.set_variation_by_name(var)
    return f

def text_sprite(txt, path, size, var="Bold", top=(255, 236, 140), bot=(255, 150, 30),
                stroke=12, depth=12, ext=(150, 40, 70), outline=(255, 255, 255)):
    f = font(path, size, var)
    pad = stroke + depth + 30
    bb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), txt, font=f, stroke_width=stroke)
    w, h = bb[2] - bb[0] + pad * 2, bb[3] - bb[1] + pad * 2
    org = (pad - bb[0], pad - bb[1])
    m_fill = Image.new("L", (w, h)); ImageDraw.Draw(m_fill).text(org, txt, font=f, fill=255)
    m_str = Image.new("L", (w, h)); ImageDraw.Draw(m_str).text(org, txt, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh = Image.new("RGBA", (w, h), (40, 0, 30, 0)); sh.putalpha(m_str.point(lambda v: v * .55).filter(ImageFilter.GaussianBlur(10)))
    out.alpha_composite(sh, (int(depth * .7) + 6, depth + 10))
    for i in range(depth, 0, -1):               # 3D extrusion
        k = 0.75 + 0.25 * (1 - i / depth)
        layer = Image.new("RGBA", (w, h), tuple(int(c * k) for c in ext) + (255,))
        tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0)); tmp.paste(layer, (int(i * .5), i), m_str)
        out.alpha_composite(tmp)
    ol = Image.new("RGBA", (w, h), outline + (255,)); tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0)); tmp.paste(ol, (0, 0), m_str)
    out.alpha_composite(tmp)
    g = np.zeros((h, w, 4), np.uint8)
    ys = np.linspace(0, 1, h)[:, None]
    for c in range(3): g[..., c] = (top[c] + (bot[c] - top[c]) * ((ys - (org[1]/h)) / max(.01, 1 - 2*org[1]/h)).clip(0, 1)).astype(np.uint8)
    g[..., 3] = 255
    gi = Image.fromarray(g, "RGBA"); tmp = Image.new("RGBA", (w, h), (0, 0, 0, 0)); tmp.paste(gi, (0, 0), m_fill)
    out.alpha_composite(tmp)
    # glossy highlight on top third of letters
    hl = Image.new("L", (w, h)); hd = ImageDraw.Draw(hl)
    hd.text((org[0], org[1] - max(3, size // 18)), txt, font=f, fill=255)
    hl = Image.fromarray((np.asarray(m_fill) * (np.asarray(hl) == 0) * 0).astype(np.uint8))  # placeholder no-op
    return out

def pill_sprite(txt, size=50):
    f = font(FONT_EN, size, "SemiBold")
    bb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), txt, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    w, h = tw + 110, th + 64
    out = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    sh = Image.new("L", out.size); ImageDraw.Draw(sh).rounded_rectangle((24, 30, w + 24, h + 30), h // 2, fill=120)
    out.alpha_composite(Image.merge("RGBA", [Image.new("L", out.size, 50)] * 3 + [sh.filter(ImageFilter.GaussianBlur(8))]))
    d = ImageDraw.Draw(out)
    d.rounded_rectangle((20, 20, w + 20, h + 20), h // 2, fill=(255, 250, 240), outline=(255, 190, 60), width=6)
    d.text((20 + 55 - bb[0], 20 + 32 - bb[1]), txt, font=f, fill=(110, 20, 70))
    return out

# ----------------------------------------------------------------- cute 3D diya
def diya_sprite(lit, scale=1.0):
    k = 3
    w, h = 260 * k, 170 * k
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx, rim_y = w // 2, 50 * k
    # soft contact shadow
    sh = Image.new("L", (w, h)); ImageDraw.Draw(sh).ellipse((cx - 110 * k, 140 * k, cx + 110 * k, 166 * k), fill=110)
    im.alpha_composite(Image.merge("RGBA", [Image.new("L", (w, h), 40)] * 3 + [sh.filter(ImageFilter.GaussianBlur(8 * k))]))
    # bowl: shaded via concentric ellipses (fake 3D)
    for i in range(40):
        t = i / 39
        col = (int(lerp(150, 245, t)), int(lerp(55, 130, t)), int(lerp(30, 70, t)))
        ox = int(lerp(0, -22 * k, t)); oy = int(lerp(0, -18 * k, t))
        rx = lerp(118, 40, t) * k; ry = lerp(95, 30, t) * k
        d.chord((cx + ox - rx, rim_y - ry + oy + 12 * k, cx + ox + rx, rim_y + ry + oy + 12 * k), 0, 180, fill=col)
    # spout
    d.polygon([(cx + 95 * k, rim_y), (cx + 128 * k, rim_y - 22 * k), (cx + 110 * k, rim_y + 18 * k)], fill=(205, 90, 45))
    # rim + oil
    d.ellipse((cx - 118 * k, rim_y - 20 * k, cx + 118 * k, rim_y + 26 * k), fill=(230, 120, 60))
    d.ellipse((cx - 104 * k, rim_y - 12 * k, cx + 104 * k, rim_y + 18 * k), fill=(120, 40, 25))
    d.ellipse((cx - 96 * k, rim_y - 7 * k, cx + 96 * k, rim_y + 15 * k), fill=(250, 190, 60))
    d.ellipse((cx - 60 * k, rim_y - 4 * k, cx + 20 * k, rim_y + 6 * k), fill=(255, 225, 130))
    # painted dots band
    for j in range(9):
        a = math.pi * (0.12 + 0.76 * j / 8)
        x = cx - math.cos(a) * 92 * k; y = rim_y + 22 * k + math.sin(a) * 18 * k
        r = 6 * k
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 230, 120) if j % 2 else (255, 255, 255))
    # kawaii face
    fy = 102 * k
    if lit:
        for ex in (-30, 30):
            d.ellipse((cx + ex * k - 9 * k, fy - 12 * k, cx + ex * k + 9 * k, fy + 9 * k), fill=(45, 15, 10))
            d.ellipse((cx + ex * k - 3 * k, fy - 8 * k, cx + ex * k + 3 * k, fy - 2 * k), fill=(255, 255, 255))
        d.arc((cx - 14 * k, fy - 2 * k, cx + 14 * k, fy + 18 * k), 15, 165, fill=(45, 15, 10), width=4 * k)
    else:
        for ex in (-30, 30):
            d.arc((cx + ex * k - 12 * k, fy - 12 * k, cx + ex * k + 12 * k, fy + 6 * k), 20, 160, fill=(45, 15, 10), width=4 * k)
        d.ellipse((cx - 5 * k, fy + 6 * k, cx + 5 * k, fy + 14 * k), fill=(45, 15, 10))
    for ex in (-52, 52):
        d.ellipse((cx + ex * k - 14 * k, fy + 4 * k, cx + ex * k + 14 * k, fy + 16 * k), fill=(255, 120, 140))
    # specular highlight
    hl = Image.new("L", (w, h)); ImageDraw.Draw(hl).ellipse((cx - 85 * k, 78 * k, cx - 55 * k, 120 * k), fill=150)
    im.alpha_composite(Image.merge("RGBA", [Image.new("L", (w, h), 255)] * 3 + [hl.filter(ImageFilter.GaussianBlur(6 * k))]))
    # wick
    d.line((cx + 70 * k, rim_y - 4 * k, cx + 92 * k, rim_y - 22 * k), fill=(250, 240, 220), width=6 * k)
    im = im.resize((int(w / k * scale), int(h / k * scale)), Image.LANCZOS)
    return im

def flame_sprite(seed, scale=1.0):
    rng = random.Random(seed)
    k = 3; w, h = 90 * k, 150 * k
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    sway = rng.uniform(-8, 8) * k; hh = rng.uniform(0.85, 1.12)
    nx = (x - w / 2) / (w * .32); ny = (y - h * .78) / (h * .62 * hh)
    nx = nx - (sway / (w * .32)) * np.clip(-ny, 0, 1) ** 2
    # teardrop field
    r = np.sqrt(nx ** 2 / np.clip(1 + ny, .05, 2) ** 1.6 + (ny * 0.95) ** 2)
    out = np.zeros((h, w, 4), np.float32)
    def lay(rad, col):
        a = np.clip((rad - r) / (0.08), 0, 1)[..., None]
        out[..., :3] = out[..., :3] * (1 - a) + np.array(col, np.float32) / 255 * a
        out[..., 3:] = np.maximum(out[..., 3:], a)
    lay(1.0, (255, 110, 20)); lay(.78, (255, 185, 40)); lay(.5, (255, 240, 150)); lay(.25, (255, 255, 240))
    im = Image.fromarray((out * 255).astype(np.uint8), "RGBA").resize((int(w / k * scale), int(h / k * scale)), Image.LANCZOS)
    return to_np(im)

# ----------------------------------------------------------------- person "paper sticker"
def person_sprite(target_w=880):
    c = Image.open(f"{S}/cut.png").convert("RGBA")
    c = c.crop(c.getbbox())
    sc = target_w / c.width
    c = c.resize((int(c.width * sc), int(c.height * sc)), Image.LANCZOS)
    a = np.asarray(c).copy()
    ext = 260   # extend bottom so the sticker edge never shows
    a = np.concatenate([a, np.repeat(a[-1:], ext, axis=0)], axis=0)
    rgb = a[..., :3].astype(np.float32); al = a[..., 3]
    # paper look: slight warm tone + grain
    rng = np.random.default_rng(3)
    grain = rng.normal(0, 5, al.shape)[..., None]
    rgb = np.clip(rgb * 1.04 + np.array([6, 3, -2]) + grain, 0, 255)
    pad = 60
    al_p = np.pad(al, pad); rgb_p = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)))
    hard = (al_p > 110).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    outl = cv2.dilate(hard, ker)
    outl = cv2.GaussianBlur(outl, (5, 5), 0).astype(np.float32) / 255
    al_f = al_p.astype(np.float32) / 255
    Hh, Ww = al_f.shape
    out = np.zeros((Hh, Ww, 4), np.float32)
    # white paper border with a tiny cream edge line
    out[..., :3] = np.array([1.0, 1.0, 0.99])
    out[..., 3] = outl
    edge = np.clip(outl - cv2.erode(outl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), 0, 1)[..., None]
    out[..., :3] = out[..., :3] * (1 - edge * .25) + np.array([.85, .8, .72]) * edge * .25
    a3 = al_f[..., None]
    out[..., :3] = out[..., :3] * (1 - a3) + rgb_p / 255 * a3
    # drop shadow canvas
    sh = cv2.GaussianBlur(outl, (0, 0), 14)
    off = (22, 26)
    fin = np.zeros((Hh + off[1] + 40, Ww + off[0] + 40, 4), np.float32)
    fin[off[1] + 20:off[1] + 20 + Hh, off[0] + 20:off[0] + 20 + Ww, :3] = np.array([.16, .02, .12])
    fin[off[1] + 20:off[1] + 20 + Hh, off[0] + 20:off[0] + 20 + Ww, 3] = sh * .55
    sub = fin[20:20 + Hh, 20:20 + Ww]
    aa = out[..., 3:4]
    sub[..., :3] = sub[..., :3] * (1 - aa) + out[..., :3] * aa
    sub[..., 3:4] = aa + sub[..., 3:4] * (1 - aa)
    return Image.fromarray((np.clip(fin, 0, 1) * 255).astype(np.uint8), "RGBA")

# ----------------------------------------------------------------- background art
def gradient(stops):
    ys = np.linspace(0, 1, H)
    out = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        out[..., c] = np.interp(ys, [s[0] for s in stops], [s[1][c] / 255 for s in stops])[:, None]
    return out

def kolam_sprite(size=1500):
    k = 2; s = size * k
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = s / 2
    col = (255, 210, 150, 255)
    for ring, (rad, n, pr, wdt) in enumerate([(160, 8, 90, 6), (330, 12, 120, 6), (520, 16, 130, 5), (690, 24, 90, 5)]):
        rad *= k; pr *= k
        for i in range(n):
            a = 2 * math.pi * i / n
            x, y = c + math.cos(a) * rad, c + math.sin(a) * rad
            # petal (lens shape)
            pts = []
            for t in np.linspace(0, math.pi, 20):
                pts.append((x + math.cos(a) * (math.cos(t) * pr / 2) - math.sin(a) * (math.sin(t) * pr / 4),
                            y + math.sin(a) * (math.cos(t) * pr / 2) + math.cos(a) * (math.sin(t) * pr / 4)))
            for t in np.linspace(math.pi, 2 * math.pi, 20):
                pts.append((x + math.cos(a) * (math.cos(t) * pr / 2) - math.sin(a) * (math.sin(t) * pr / 4),
                            y + math.sin(a) * (math.cos(t) * pr / 2) + math.cos(a) * (math.sin(t) * pr / 4)))
            d.line(pts + [pts[0]], fill=col, width=wdt * k)
            dx, dy = c + math.cos(a + math.pi / n) * rad, c + math.sin(a + math.pi / n) * rad
            d.ellipse((dx - 9 * k, dy - 9 * k, dx + 9 * k, dy + 9 * k), fill=col)
        d.ellipse((c - rad + pr * .7, c - rad + pr * .7, c + rad - pr * .7, c + rad - pr * .7), outline=col, width=3 * k) if ring else None
    d.ellipse((c - 50 * k, c - 50 * k, c + 50 * k, c + 50 * k), outline=col, width=8 * k)
    return im.resize((size, size), Image.LANCZOS)

def toran_sprite():
    k = 2; w, h = W * k, 200 * k
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    n = 15
    for i in range(n):
        x = (i + .5) * w / n
        sag = 40 * k * math.sin(math.pi * ((i % 5) + .5) / 5)
        y = 30 * k + sag
        # mango leaf hanging
        lp = [(x, y), (x - 16 * k, y + 50 * k), (x, y + 95 * k), (x + 16 * k, y + 50 * k)]
        d.polygon(lp, fill=(40, 140, 60)); d.line((x, y, x, y + 90 * k), fill=(90, 190, 90), width=2 * k)
        for j, cc in enumerate([(255, 140, 0), (255, 200, 30)]):
            xx = x + (w / n / 2)
            yy = 30 * k + 40 * k * math.sin(math.pi * (((i % 5) + 1) % 5 + .0) / 5) * 0 + sag + 6 * k
            r = 26 * k
            d.ellipse((xx - r, yy - r, xx + r, yy + r), fill=(230, 110, 0) if (i % 2) else (240, 175, 0))
            for p in range(10):
                a = p * 2 * math.pi / 10
                px, py = xx + math.cos(a) * r * .55, yy + math.sin(a) * r * .55
                d.ellipse((px - 9 * k, py - 9 * k, px + 9 * k, py + 9 * k), fill=(255, 160, 20) if (i % 2) else (255, 215, 60))
            d.ellipse((xx - 8 * k, yy - 8 * k, xx + 8 * k, yy + 8 * k), fill=(200, 80, 0))
            break
    d.line([(0, 30 * k)] + [((i + .5) * w / n, 30 * k + 40 * k * math.sin(math.pi * ((i % 5) + .5) / 5)) for i in range(n)] + [(w, 30 * k)], fill=(255, 220, 120), width=3 * k)
    return im.resize((W, 200), Image.LANCZOS)

def star_sprite(size=80, col=(255, 245, 200)):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    c = size / 2; dx, dy = np.abs(x - c) / c, np.abs(y - c) / c
    a = np.clip(1 - (np.sqrt(dx) + np.sqrt(dy)), 0, 1) ** 1.2
    a = np.maximum(a, np.clip(1 - np.sqrt(dx * dx + dy * dy) * 4, 0, 1))
    out = np.zeros((size, size, 4), np.float32); out[..., :3] = np.array(col) / 255; out[..., 3] = a
    return out

def petal_sprites():
    out = []
    for i in range(12):
        im = Image.new("RGBA", (60, 60)); d = ImageDraw.Draw(im)
        col = [(255, 140, 0), (255, 190, 20), (230, 60, 40)][i % 3]
        d.ellipse((18, 6, 42, 54), fill=col + (255,)); d.ellipse((24, 12, 32, 34), fill=(255, 230, 150, 140))
        out.append(to_np(im.rotate(i * 30, resample=Image.BICUBIC)))
    return out

# ----------------------------------------------------------------- TIMELINE (shared with audio)
EV = {}  # name -> time

def build_assets():
    A = {}
    A["bg_dark"] = gradient([(0, (8, 6, 30)), (.6, (25, 10, 55)), (1, (45, 12, 60))])
    A["bg_warm"] = gradient([(0, (95, 10, 75)), (.35, (175, 25, 85)), (.7, (235, 85, 45)), (1, (255, 160, 50))])
    A["kolam"] = kolam_sprite()
    A["toran"] = to_np(toran_sprite())
    A["diya_off"] = diya_sprite(False, 1.0)
    A["diya_on"] = diya_sprite(True, 1.0)
    A["diya_off_s"] = diya_sprite(False, .62)
    A["diya_on_s"] = diya_sprite(True, .62)
    A["flames"] = [flame_sprite(i, 1.0) for i in range(8)]
    A["flames_s"] = [flame_sprite(i, .62) for i in range(8)]
    A["glow"] = radial(520, (255, 170, 60), 2.2, .9)
    A["glow_s"] = radial(300, (255, 170, 60), 2.2, .8)
    A["bokeh"] = [radial(s, c, 1.3, .5) for s, c in [(60, (255, 210, 120)), (90, (255, 150, 90)), (40, (255, 240, 200)), (120, (255, 120, 160))]]
    A["star"] = star_sprite(90); A["star_s"] = star_sprite(44)
    A["dot"] = radial(38, (255, 230, 160), 1.2, 1.0)
    A["dot2"] = radial(38, (255, 120, 200), 1.2, 1.0)
    A["dot3"] = radial(38, (140, 255, 200), 1.2, 1.0)
    A["petals"] = petal_sprites()
    A["person"] = person_sprite()
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
    T["translit"] = text_sprite("Deepavali Nalvazhthukal", FONT_EN, 46, var="SemiBold", top=(255,255,255), bot=(255,240,220), stroke=0, depth=4, ext=(90,20,60))
    T["w1"] = text_sprite("May the light of Deepavali", FONT_EN, 70, var="SemiBold", **white, stroke=8, depth=6)
    T["w2"] = text_sprite("fill your home with", FONT_EN, 70, var="SemiBold", **white, stroke=8, depth=6)
    T["w3"] = text_sprite("JOY, PEACE", FONT_EN, 130, **gold, stroke=14, depth=14)
    T["w4"] = text_sprite("& PROSPERITY", FONT_EN, 112, **pink, stroke=14, depth=12)
    T["o1"] = text_sprite("Happy Deepavali!", FONT_EN, 112, **gold, stroke=12, depth=12)
    T["o2"] = text_sprite("தீபாவளி நல்வாழ்த்துகள்!", FONT_TA, 74, var="ExtraBold", **white, stroke=10, depth=8)
    T["o3"] = text_sprite("Warm wishes from", FONT_EN, 52, var="Medium", top=(255,255,255), bot=(255,240,220), stroke=0, depth=4, ext=(90,20,60))
    T["sign"] = pill_sprite("W/LL/A///.  |  williamsoon.com", 46)
    A["T"] = T
    return A

# text elements: key, y, t_in, t_out, style
TEXT = [
    ("one",   470, 0.25, 1.72, "pop"),
    ("tiny",  620, 0.65, 1.72, "pop"),
    ("light", 790, 1.05, 1.72, "pop"),
    ("can",   480, 2.00, 3.00, "pop"),
    ("every", 640, 2.50, 3.00, "slam"),
    ("happy", 250, 5.00, 9.00, "drop"),
    ("deep",  430, 5.50, 9.00, "slam"),
    ("ta1",   610, 6.50, 9.00, "pop"),
    ("ta2",   745, 7.00, 9.00, "pop"),
    ("translit", 850, 7.50, 9.00, "fade"),
    ("w1",    240, 9.25, 12.20, "pop"),
    ("w2",    340, 9.75, 12.20, "pop"),
    ("w3",    490, 10.25, 12.20, "slam"),
    ("w4",    640, 10.75, 12.20, "slam"),
    ("o1",    250, 12.50, 99, "drop"),
    ("o2",    390, 13.00, 99, "pop"),
    ("o3",    500, 13.50, 99, "fade"),
    ("sign",  600, 13.75, 99, "pop"),
]
FIREWORKS = [(8.0, 230, 330, 0), (8.5, 850, 260, 1), (11.0, 180, 200, 2), (13.0, 540, 180, 0), (13.25, 880, 330, 2), (13.4, 210, 360, 1)]
DIYA_X = [140, 340, 540, 740, 940]
DIYA_LIT = [3.75, 4.0, 4.25, 4.5, 4.75]     # row lights one by one (centre one is the hook diya)
DIYA_ORDER = [0, 1, 3, 4]                   # index 2 = hook diya, already lit
HOP = [(7.5, -80), (8.5, 80), (10.0, -45), (11.5, 0), (12.5, 0)]  # (time, new x)
SLAMS = [t for k, y, t, o, s in TEXT if s == "slam"]

def text_anim(t, t0, t1, style):
    """returns (scale, angle, dy, opacity)"""
    if t < t0 or t > t1 + 0.3: return None
    u = t - t0
    if style == "pop":
        sc = ease_out_back(u / 0.35, 2.4); ang = (1 - ease_out_cubic(u / .35)) * -12; dy = 0; op = clamp(u / .1)
    elif style == "slam":
        sc = lerp(2.6, 1.0, ease_out_cubic(u / 0.18)); ang = 0; dy = 0; op = clamp(u / .08)
        if u > .18: sc = 1 + 0.06 * math.exp(-(u - .18) * 12) * math.sin((u - .18) * 40)
    elif style == "drop":
        sc = 1; ang = (1 - ease_out_cubic(u / .5)) * 8; dy = -500 * (1 - ease_out_bounce(u / 0.55)); op = clamp(u / .1)
    else:
        sc = lerp(.9, 1, ease_out_cubic(u / .4)); ang = 0; dy = 20 * (1 - ease_out_cubic(u / .4)); op = clamp(u / .4)
    sc *= 1 + 0.015 * math.sin(t * 5 + t0 * 3)  # breathing
    if t > t1:  # exit: shrink + fly up
        v = (t - t1) / 0.3
        sc *= 1 - ease_in_cubic(v); dy -= 120 * ease_in_cubic(v); op *= 1 - v
    return sc, ang, dy, op

def person_state(t):
    """stop-motion: pose updates at 12 fps"""
    tq = math.floor(t * 12) / 12
    rng = random.Random(int(tq * 12) * 7 + 1)
    if tq < 3.0: return None
    base_y = 820  # top of sprite region (approx); we compute centre later
    u = tq - 3.0
    rise = 1 - ease_out_back(u / 0.5, 2.0)
    y = 1300 * rise
    x = 0; px = 0; hop_y = 0; sq = 1.0; tilt = 0
    for (th, nx) in HOP:
        if tq >= th:
            v = (tq - th) / 0.42
            if v < 1:
                x = lerp(px, nx, ease_io(v)); hop_y = -110 * math.sin(math.pi * v); tilt = (nx - px) * 0.06 * math.sin(math.pi * v)
            else:
                x = nx
                land = tq - th - 0.42
                if land < 0.2: sq = 1 - 0.05 * math.sin(math.pi * land / 0.2)
            px = nx
        else:
            if x == 0 and px == 0: pass
            break
    # beat bob from 5s
    bob = 0
    if tq > 3.6: bob = -12 * abs(math.sin(math.pi * (tq - 3.5) * 2))
    jit_r = rng.uniform(-1.6, 1.6); jit_y = rng.uniform(-4, 4)
    return dict(x=x, y=y + hop_y + bob + jit_y, ang=tilt + jit_r + 2.5 * math.sin(tq * 2.2), sq=sq)

def firework_particles(t, ft, fx, fy, kind):
    if t < ft - 0.55 or t > ft + 1.6: return [], None
    if t < ft:  # rocket rising
        v = (t - (ft - 0.55)) / 0.55
        return [], (fx, lerp(1500, fy, ease_out_cubic(v)))
    rng = random.Random(int(ft * 100))
    u = t - ft; pts = []
    n = 70
    for i in range(n):
        a = 2 * math.pi * i / n + rng.uniform(-.05, .05)
        sp = rng.uniform(260, 420)
        dist = sp * (1 - math.exp(-u * 3.2)) / 1.0
        x = fx + math.cos(a) * dist; y = fy + math.sin(a) * dist + 120 * u * u
        life = clamp(1 - u / 1.5)
        tw = 0.6 + 0.4 * math.sin(u * 40 + i)
        pts.append((x, y, life * tw))
    return pts, None

def render_frame(fi):
    t = fi / FPS
    A = ASSETS
    # ---- background: dark → warm reveal through a circle from the hook diya at 2.0s
    reveal = ease_out_cubic((t - 2.0) / 0.6)
    if t < 2.0: frame = A["bg_dark"].copy()
    elif reveal >= 1: frame = A["bg_warm"].copy()
    else:
        yy, xx = np.ogrid[0:H, 0:W]
        rr = reveal * 2300
        m = np.clip((rr - np.sqrt((xx - 540) ** 2 + (yy - 1250) ** 2)) / 60, 0, 1)[..., None].astype(np.float32)
        frame = A["bg_dark"] * (1 - m) + A["bg_warm"] * m
    # kolam slowly turning
    if t >= 2.0:
        k = A["kolam"].rotate(-t * 8, resample=Image.BILINEAR)
        blit(frame, to_np(k), 540, 1080, 0.18 * reveal)
    # floating bokeh
    rng = random.Random(5)
    for i in range(26):
        sp = A["bokeh"][i % 4]
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H); vs = rng.uniform(20, 60); ph = rng.uniform(0, 6)
        y = (y0 - vs * t) % (H + 200) - 100; x = x0 + 30 * math.sin(t * .8 + ph)
        op = (0.25 if t < 2 else 0.55) * (0.6 + 0.4 * math.sin(t * 2 + ph))
        blit(frame, sp, x, y, op, add=True)
    # toran at top (after reveal)
    if t >= 2.2:
        v = ease_out_bounce((t - 2.2) / 0.6)
        blit(frame, A["toran"], 540 + 4 * math.sin(t * 1.7), lerp(-120, 100, v), 1.0)
    # ---- person sticker
    ps = person_state(t)
    if ps:
        P = A["person"]
        spr = xform(P, 1.0, ps["ang"])
        if ps["sq"] != 1.0:
            spr = to_np(Image.fromarray((spr * 255).astype(np.uint8)).resize((int(spr.shape[1] * (2 - ps["sq"])), int(spr.shape[0] * ps["sq"]))))
        cy = 905 + spr.shape[0] / 2 - 30 + ps["y"]
        blit(frame, spr, 540 + ps["x"], cy, 1.0)
    # ---- diyas
    stop = math.floor(t * 12) / 12
    if t < 3.0:  # hook diya, centre, big
        dx, dy = 540, 1250
        lit = t >= 1.75
        blit(frame, to_np(A["diya_on"] if lit else A["diya_off"]), dx, dy, clamp(t / 0.3))
        if lit:
            g = clamp((t - 1.75) / 0.25)
            blit(frame, A["glow"], dx + 80, dy - 90, g * (0.9 + 0.1 * math.sin(t * 20)), add=True)
            fl = A["flames"][int(stop * 12) % 8]
            blit(frame, fl, dx + 92, dy - 115 - 10 * (1 - g), g)
        if 1.5 <= t < 1.8:  # match spark
            for j in range(10):
                r = random.Random(fi * 10 + j)
                blit(frame, A["dot"], dx + 92 + r.uniform(-40, 40), dy - 100 + r.uniform(-40, 30), 1.0, add=True)
    if t >= 3.0:
        for i, x in enumerate(DIYA_X):
            enter = ease_out_back((t - 3.0 - i * 0.06) / 0.4)
            if enter <= 0: continue
            y = lerp(2100, 1790, enter) + 6 * math.sin(stop * 6 + i)
            lit_t = 1.75 if i == 2 else DIYA_LIT[DIYA_ORDER.index(i)] if i in DIYA_ORDER else 99
            lit = t >= lit_t
            sq = 1.0
            if lit and t - lit_t < 0.25: sq = 1 + 0.15 * math.sin(math.pi * (t - lit_t) / 0.25)
            d = A["diya_on_s"] if lit else A["diya_off_s"]
            spr = to_np(d) if sq == 1 else to_np(d.resize((int(d.width * (2 - sq)), int(d.height * sq))))
            blit(frame, spr, x, y - (sq - 1) * 40, 1.0)
            if lit:
                blit(frame, A["glow_s"], x + 50, y - 60, 0.85 + 0.15 * math.sin(t * 17 + i), add=True)
                blit(frame, A["flames_s"][(int(stop * 12) + i * 3) % 8], x + 57, y - 72, 1.0)
    # ---- petals falling after the big greeting
    if t >= 5.5:
        r = random.Random(11)
        for i in range(30):
            x0 = r.uniform(0, W); st = 5.5 + r.uniform(0, 3); vs = r.uniform(180, 320); ph = r.uniform(0, 6)
            if t < st: continue
            y = -40 + (t - st) * vs
            if y > H + 40: continue
            x = x0 + 50 * math.sin((t - st) * 2 + ph)
            blit(frame, A["petals"][int((t * 6 + i)) % 12], x, y, 0.95)
    # ---- fireworks
    for (ft, fx, fy, kind) in FIREWORKS:
        pts, rocket = firework_particles(t, ft, fx, fy, kind)
        dot = [A["dot"], A["dot2"], A["dot3"]][kind]
        if rocket: blit(frame, A["dot"], rocket[0], rocket[1], 1.0, add=True)
        for (x, y, op) in pts: blit(frame, dot, x, y, op * 1.4, add=True)
        if 0 <= t - ft < 0.12: blit(frame, A["glow"], fx, fy, 1 - (t - ft) / .12, add=True)
    # ---- kinetic text
    T = A["T"]
    for key, y, t0, t1, style in TEXT:
        st = text_anim(t, t0, t1, style)
        if not st: continue
        sc, ang, dy, op = st
        spr = xform(T[key], sc, ang)
        if spr is not None: blit(frame, spr, 540, y + dy, op)
    # sparkles twinkling around titles
    if t >= 2.0:
        r = random.Random(21)
        for i in range(14):
            x = r.uniform(60, W - 60); y = r.uniform(140, 900); ph = r.uniform(0, 6); sp = r.uniform(3, 6)
            s = max(0, math.sin(t * sp + ph)) ** 3
            if s > 0.05: blit(frame, A["star_s"] if i % 2 else A["star"], x, y, s, add=True)
    # flash at the bloom + at the big DEEPAVALI slam
    for ft, strength in [(2.0, 0.85), (5.5, 0.35), (13.0, 0.25)]:
        if 0 <= t - ft < 0.25: frame += strength * (1 - (t - ft) / 0.25)
    # vignette + shake
    frame *= VIG
    for ts in SLAMS + [2.0]:
        if 0 <= t - ts < 0.2:
            a = 14 * (1 - (t - ts) / 0.2)
            frame = np.roll(frame, (int(a * math.sin(fi * 2.1)), int(a * math.cos(fi * 1.7))), axis=(0, 1))
    if t > DUR - 0.4: frame *= clamp((DUR - t) / 0.4) * 0.5 + 0.5
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8).tobytes()

def make_vig():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * .75)) ** 2 + ((yy - H / 2) / (H * .7)) ** 2)
    return (1 - 0.35 * np.clip(d - 0.4, 0, 1) ** 1.5)[..., None]

ASSETS = None; VIG = None
def init():
    global ASSETS, VIG
    ASSETS = build_assets(); VIG = make_vig()

if __name__ == "__main__":
    init()
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        for tt in [float(x) for x in sys.argv[2:]]:
            Image.frombytes("RGB", (W, H), render_frame(int(tt * FPS))).save(f"{S}/still_{tt:05.2f}.jpg", quality=88)
        sys.exit()
    from multiprocessing import Pool
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", f"{S}/video_noaudio.mp4"],
                          stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for i, b in enumerate(pool.imap(render_frame, range(NF), chunksize=4)):
            ff.stdin.write(b)
            if i % 60 == 0: print("frame", i, flush=True)
    ff.stdin.close(); ff.wait()
    print("done")
