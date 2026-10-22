# -*- coding: utf-8 -*-
"""
GENERADOR DE VIDEOS DE MISTERIO (prototipo)  -  "El hombre que aparecía en todas las fotografías"
Todo se dibuja y se anima por código: no usa IA de imágenes, así que el hombre
misterioso y la familia son IDÉNTICOS en todas las escenas.
Salida: MP4 vertical 720x1280, 30 fps, con voz, música, efectos y subtítulos selectivos.
"""
import os, sys, math, subprocess, wave, time, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

# =====================================================================
# 1. CONFIGURACIÓN  (aquí es donde cambias cosas)
# =====================================================================
TITULO = "El hombre que aparecía en todas las fotografías"
SALIDA = "el_hombre_fotografias.mp4"
VOZ = "es-MX-JorgeNeural"                        # voz latina masculina
VOCES_ALTERNAS = ["es-MX-DaliaNeural", "es-US-AlonsoNeural"]   # si la primera falla
TONO = "-4Hz"                                    # más grave = más misterioso
VELOCIDAD_BASE = -4                              # % (negativo = más lento)
DURACION_OBJETIVO = (46.0, 54.5)                 # el video debe quedar entre estos segundos
ETIQUETA_FICCION = "HISTORIA DE FICCIÓN"         # aviso discreto al inicio ("" para quitarlo)

# (clave, texto narrado, pausa antes en segundos)
LINEAS = [
    ("s1a", "Esta parece una foto familiar normal.", 0.0),
    ("s1b", "Pero mira bien… al fondo.", 0.30),
    ("s2a", "Un día, la familia lo notó: un hombre que nadie recordaba haber visto.", 0.35),
    ("s2b", "Y no aparecía solo en esa foto.", 0.30),
    ("s3a", "Revisaron imágenes de años distintos.", 0.35),
    ("s3b", "En mil novecientos noventa y ocho, al otro lado de la calle.", 0.30),
    ("s3c", "En dos mil cuatro, junto a la puerta de su casa.", 0.30),
    ("s4a", "En dos mil once… justo detrás de la abuela.", 0.35),
    ("s4b", "Siempre el mismo hombre. Siempre un poco más cerca.", 0.30),
    ("s4c", "Entonces, la familia revisó la última foto que habían tomado.", 0.35),
    ("s5a", "El hombre ya no estaba al fondo.", 0.45),
    ("s5b", "Estaba mirando… desde la ventana de su propia casa.", 1.75),
    ("s5c", "Y la foto… era de hoy.", 0.55),
]

W, H, FPS = 720, 1280, 30
SR = 44100
WORK = "trabajo"

RS = getattr(Image, "Resampling", Image)
LANCZOS, BICUBIC, BILINEAR = RS.LANCZOS, RS.BICUBIC, RS.BILINEAR
AFFINE = getattr(getattr(Image, "Transform", Image), "AFFINE")


# =====================================================================
# 2. UTILIDADES
# =====================================================================
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, fin=0.3, fout=0.3):
    if t < t0 or t > t1:
        return 0.0
    return clamp((t - t0) / max(fin, 1e-6)) * clamp((t1 - t) / max(fout, 1e-6))


def _fontdir():
    cands = []
    try:
        import matplotlib
        cands.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf"))
    except Exception:
        pass
    cands.append("/usr/share/fonts/truetype/dejavu")
    for c in cands:
        if os.path.exists(os.path.join(c, "DejaVuSans-Bold.ttf")):
            return c
    return None


_FD = _fontdir()
_FC = {}


def F(size, mono=False, bold=False):
    key = (int(size), mono, bold)
    if key not in _FC:
        name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
        try:
            _FC[key] = ImageFont.truetype(os.path.join(_FD, name), int(size))
        except Exception:
            _FC[key] = ImageFont.load_default()
    return _FC[key]


# =====================================================================
# 3. DIBUJO VECTORIAL (todo en "unidades de foto")
# =====================================================================
class Cv:
    def __init__(s, w, h, k):
        s.w, s.h, s.k = w, h, k
        s.im = Image.new("RGB", (int(round(w * k)), int(round(h * k))), (0, 0, 0))
        s.d = ImageDraw.Draw(s.im)

    def _p(s, pts):
        return [(x * s.k, y * s.k) for x, y in pts]

    def rect(s, x0, y0, x1, y1, f=None, o=None, w=1, r=0):
        b = [x0 * s.k, y0 * s.k, x1 * s.k, y1 * s.k]
        if r:
            rr = min(r * s.k, (b[2] - b[0]) / 2, (b[3] - b[1]) / 2)
            s.d.rounded_rectangle(b, radius=max(0, rr), fill=f, outline=o, width=max(1, int(w * s.k)))
        else:
            s.d.rectangle(b, fill=f, outline=o, width=max(1, int(w * s.k)))

    def ell(s, cx, cy, rx, ry=None, f=None, o=None, w=1):
        ry = rx if ry is None else ry
        s.d.ellipse([(cx - rx) * s.k, (cy - ry) * s.k, (cx + rx) * s.k, (cy + ry) * s.k],
                    fill=f, outline=o, width=max(1, int(w * s.k)))

    def poly(s, pts, f=None):
        s.d.polygon(s._p(pts), fill=f)

    def line(s, pts, f, w=1):
        s.d.line(s._p(pts), fill=f, width=max(1, int(w * s.k)), joint="curve")

    def vgrad(s, x0, y0, x1, y1, c0, c1):
        hh = max(1, int((y1 - y0) * s.k))
        ww = max(1, int((x1 - x0) * s.k))
        t = np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]
        a = np.array(c0, np.float32)[None, None, :] * (1 - t) + np.array(c1, np.float32)[None, None, :] * t
        a = np.repeat(a, ww, axis=1).astype(np.uint8)
        s.im.paste(Image.fromarray(a), (int(x0 * s.k), int(y0 * s.k)))


def dim(col, f):
    return tuple(max(0, min(255, int(v * f))) for v in col)


# ---- El hombre misterioso: SIEMPRE los mismos colores y rasgos ----
M_SKIN = (198, 190, 178)
M_HAIR = (22, 20, 24)
M_COAT = (34, 36, 44)
M_COAT2 = (58, 61, 74)
M_PANTS = (22, 22, 28)


def man_head(c, cx, cy, hr, lum=1.0):
    S = lambda col: dim(col, lum)
    c.ell(cx, cy - hr * 0.10, hr * 1.0, hr * 1.02, f=S(M_HAIR))                    # pelo corto
    c.ell(cx - hr * 0.86, cy + hr * 0.15, hr * 0.14, hr * 0.22, f=S(M_SKIN))        # orejas
    c.ell(cx + hr * 0.86, cy + hr * 0.15, hr * 0.14, hr * 0.22, f=S(M_SKIN))
    c.ell(cx, cy + hr * 0.12, hr * 0.84, hr * 0.97, f=S(M_SKIN))                    # rostro pálido
    c.ell(cx - hr * 0.50, cy + hr * 0.50, hr * 0.22, hr * 0.26, f=S(dim(M_SKIN, 0.82)))   # mejillas hundidas
    c.ell(cx + hr * 0.50, cy + hr * 0.50, hr * 0.22, hr * 0.26, f=S(dim(M_SKIN, 0.82)))
    c.ell(cx - hr * 0.36, cy + hr * 0.05, hr * 0.17, hr * 0.24, f=(7, 7, 9))         # ojos huecos
    c.ell(cx + hr * 0.36, cy + hr * 0.05, hr * 0.17, hr * 0.24, f=(7, 7, 9))
    c.line([(cx - hr * 0.58, cy - hr * 0.28), (cx - hr * 0.16, cy - hr * 0.22)], (20, 18, 20), hr * 0.07)
    c.line([(cx + hr * 0.58, cy - hr * 0.28), (cx + hr * 0.16, cy - hr * 0.22)], (20, 18, 20), hr * 0.07)
    c.line([(cx, cy + hr * 0.18), (cx - hr * 0.05, cy + hr * 0.42)], S(dim(M_SKIN, 0.7)), hr * 0.06)
    c.line([(cx - hr * 0.24, cy + hr * 0.64), (cx + hr * 0.24, cy + hr * 0.64)], (70, 60, 58), hr * 0.07)


def man(c, cx, by, h, lum=1.0):
    S = lambda col: dim(col, lum)
    hr = h * 0.085
    ytop = by - h + 2 * hr * 0.95
    c.rect(cx - h * .075, by - h * .42, cx - h * .005, by, f=S(M_PANTS))
    c.rect(cx + h * .005, by - h * .42, cx + h * .075, by, f=S(M_PANTS))
    c.poly([(cx - h * .15, ytop), (cx + h * .15, ytop), (cx + h * .20, by - h * .26), (cx - h * .20, by - h * .26)], S(M_COAT))
    c.poly([(cx - h * .15, ytop), (cx - h * .20, ytop + h * .03), (cx - h * .25, by - h * .38), (cx - h * .19, by - h * .38)], S(M_COAT))
    c.poly([(cx + h * .15, ytop), (cx + h * .20, ytop + h * .03), (cx + h * .25, by - h * .38), (cx + h * .19, by - h * .38)], S(M_COAT))
    c.ell(cx - h * .225, by - h * .36, h * .022, h * .03, f=S(M_SKIN))
    c.ell(cx + h * .225, by - h * .36, h * .022, h * .03, f=S(M_SKIN))
    c.rect(cx - hr * .35, ytop - hr * .5, cx + hr * .35, ytop + hr * .1, f=S(dim(M_SKIN, .8)))
    c.poly([(cx - h * .15, ytop), (cx, ytop + h * .17), (cx - h * .03, ytop)], S(M_COAT2))
    c.poly([(cx + h * .15, ytop), (cx, ytop + h * .17), (cx + h * .03, ytop)], S(M_COAT2))
    man_head(c, cx, by - h + hr, hr, lum)


def man_bust(c, cx, cy, hr, lum=1.0):
    S = lambda col: dim(col, lum)
    c.poly([(cx - hr * 2.1, cy + hr * 2.6), (cx - hr * 1.8, cy + hr * 1.35), (cx - hr * .7, cy + hr * 1.0),
            (cx + hr * .7, cy + hr * 1.0), (cx + hr * 1.8, cy + hr * 1.35), (cx + hr * 2.1, cy + hr * 2.6)], S(M_COAT))
    c.poly([(cx - hr * .7, cy + hr * 1.0), (cx, cy + hr * 1.9), (cx - hr * .2, cy + hr * 1.0)], S(M_COAT2))
    c.poly([(cx + hr * .7, cy + hr * 1.0), (cx, cy + hr * 1.9), (cx + hr * .2, cy + hr * 1.0)], S(M_COAT2))
    c.rect(cx - hr * .32, cy + hr * .7, cx + hr * .32, cy + hr * 1.2, f=S(dim(M_SKIN, .8)))
    man_head(c, cx, cy, hr, lum)


PERSONAS = {
    "abuela": dict(skin=(226, 190, 160), hair=(208, 208, 214), top=(150, 118, 172), bot=(72, 62, 92), bun=True),
    "mama": dict(skin=(214, 170, 140), hair=(62, 38, 28), top=(214, 160, 58), bot=(52, 62, 92), long=True),
    "papa": dict(skin=(222, 180, 148), hair=(42, 30, 26), top=(62, 102, 152), bot=(56, 56, 66)),
    "nino": dict(skin=(230, 188, 156), hair=(124, 82, 42), top=(222, 102, 50), bot=(60, 72, 112)),
}


def persona(c, cx, by, h, kind):
    P = PERSONAS[kind]
    kid = kind == "nino"
    hr = h * (0.125 if kid else 0.092)
    ytop = by - h + 2 * hr * 0.95
    tw = h * (0.15 if kid else 0.16)
    lw = h * 0.07
    if P.get("long"):
        c.rect(cx - hr * 1.0, by - h + hr * 0.6, cx + hr * 1.0, ytop + h * 0.2, f=P["hair"], r=hr * .4)
    c.rect(cx - lw * 1.2, by - h * .42, cx - lw * .08, by, f=P["bot"])
    c.rect(cx + lw * .08, by - h * .42, cx + lw * 1.2, by, f=P["bot"])
    c.rect(cx - lw * 1.3, by - h * .03, cx - lw * .02, by, f=(30, 28, 30))
    c.rect(cx + lw * .02, by - h * .03, cx + lw * 1.3, by, f=(30, 28, 30))
    c.rect(cx - tw, ytop, cx + tw, by - h * .40, f=P["top"], r=h * .035)
    c.rect(cx - tw - h * .05, ytop + h * .015, cx - tw + h * .012, by - h * .47, f=P["top"], r=h * .025)
    c.rect(cx + tw - h * .012, ytop + h * .015, cx + tw + h * .05, by - h * .47, f=P["top"], r=h * .025)
    c.ell(cx - tw - h * .02, by - h * .46, h * .022, f=P["skin"])
    c.ell(cx + tw + h * .02, by - h * .46, h * .022, f=P["skin"])
    c.rect(cx - hr * .3, ytop - hr * .5, cx + hr * .3, ytop + hr * .1, f=dim(P["skin"], .9))
    hy = by - h + hr
    if P.get("bun"):
        c.ell(cx, hy - hr * 1.05, hr * .42, f=P["hair"])
    c.ell(cx, hy - hr * .12, hr * 1.0, hr * 1.0, f=P["hair"])
    c.ell(cx, hy + hr * .12, hr * .84, hr * .94, f=P["skin"])
    c.ell(cx - hr * .33, hy + hr * .08, hr * .09, f=(30, 24, 22))
    c.ell(cx + hr * .33, hy + hr * .08, hr * .09, f=(30, 24, 22))
    c.line([(cx - hr * .25, hy + hr * .55), (cx, hy + hr * .64), (cx + hr * .25, hy + hr * .55)], (150, 80, 70), hr * .07)


PAL_DAY = dict(wall=(206, 190, 160), wall2=(194, 178, 150), roof=(122, 72, 62), trim=(236, 229, 214),
               door=(112, 64, 46), glass=(112, 142, 164), lit=(242, 208, 134),
               grass0=(106, 142, 72), grass1=(78, 112, 52), path=(192, 186, 174))
PAL_DUSK = dict(wall=(86, 88, 116), wall2=(76, 78, 104), roof=(40, 34, 50), trim=(150, 150, 170),
                door=(62, 40, 38), glass=(22, 28, 48), lit=(252, 204, 122),
                grass0=(40, 64, 60), grass1=(28, 48, 46), path=(100, 102, 114))


def ventana(c, x, y, w, h, pal, lit=False, bars=True, glass=None):
    c.rect(x - 5, y - 5, x + w + 5, y + h + 5, f=pal["trim"])
    c.rect(x, y, x + w, y + h, f=glass or (pal["lit"] if lit else pal["glass"]))
    if bars:
        c.line([(x + w / 2, y), (x + w / 2, y + h)], pal["trim"], 2.4)
        c.line([(x, y + h / 2), (x + w, y + h / 2)], pal["trim"], 2.4)
    c.rect(x - 9, y + h + 5, x + w + 9, y + h + 11, f=pal["trim"])


WIN = (390, 105, 70, 80)           # ventana de arriba donde aparecerá el hombre (x, y, ancho, alto)
DOOR = (394, 224, 60, 112)


def casa(c, pal, puerta_abierta=False):
    gy = 336
    c.rect(0, 60, PW, gy, f=pal["wall"])
    for yy in range(70, gy, 15):
        c.line([(0, yy), (PW, yy)], pal["wall2"], 1.2)
    c.poly([(-30, 72), (280, -50), (590, 72)], pal["roof"])
    c.rect(-30, 62, PW + 30, 76, f=pal["trim"])
    ventana(c, 92, 106, 70, 80, pal)
    ventana(c, 90, 214, 92, 72, pal, lit=(pal is PAL_DUSK))
    x, y, w, h = DOOR
    c.rect(x - 7, y - 7, x + w + 7, y + h + 4, f=pal["trim"])
    if puerta_abierta:
        c.rect(x, y, x + w, y + h, f=(12, 10, 14))
        c.poly([(x, y), (x + 14, y + 6), (x + 14, y + h - 4), (x, y + h)], pal["door"])
    else:
        c.rect(x, y, x + w, y + h, f=pal["door"])
        c.rect(x + 9, y + 10, x + w - 9, y + 46, o=dim(pal["door"], .75), w=2)
        c.rect(x + 9, y + 58, x + w - 9, y + h - 10, o=dim(pal["door"], .75), w=2)
        c.ell(x + w - 9, y + h * .52, 3.2, f=(200, 170, 80))
    c.rect(x - 18, gy - 4, x + w + 18, gy + 6, f=pal["path"])
    c.vgrad(0, gy + 6, PW, PH, pal["grass0"], pal["grass1"])
    c.poly([(x - 6, gy + 6), (x + w + 6, gy + 6), (x + w + 70, PH), (x - 70, PH)], pal["path"])


def foto_1998(c):
    c.vgrad(0, 0, PW, 205, (166, 198, 228), (228, 228, 214))
    rng = np.random.default_rng(11)
    cols = [(190, 160, 130), (170, 180, 160), (200, 186, 150), (160, 150, 170), (196, 150, 130), (150, 176, 190)]
    x = -20
    while x < PW:
        w = int(rng.integers(78, 110)); hh = int(rng.integers(52, 80))
        c.ell(x + w / 2, 150, w * .55, 36, f=(58, 96, 62))
        col = cols[int(rng.integers(0, len(cols)))]
        c.rect(x + 6, 205 - hh, x + w - 6, 205, f=col)
        c.poly([(x, 205 - hh), (x + w / 2, 205 - hh - 24), (x + w, 205 - hh)], dim(col, .6))
        c.rect(x + 16, 205 - hh + 14, x + 32, 205 - hh + 30, f=(235, 220, 160))
        c.rect(x + w - 34, 205 - hh + 14, x + w - 18, 205 - hh + 30, f=(120, 140, 160))
        x += w + int(rng.integers(0, 14))
    c.rect(0, 202, PW, 211, f=(172, 168, 158))
    c.rect(0, 211, PW, 264, f=(84, 86, 90))
    for xx in range(0, PW, 60):
        c.rect(xx, 236, xx + 30, 239, f=(220, 200, 120))
    c.rect(0, 264, PW, 282, f=(178, 174, 164))
    c.vgrad(0, 282, PW, PH, (108, 146, 74), (76, 110, 52))
    c.rect(30, 245, 38, 360, f=(120, 96, 70))
    c.rect(18, 232, 54, 252, f=(60, 80, 130), r=5)
    man(c, 455, 209, 27)
    persona(c, 150, 398, 150, "abuela")
    persona(c, 218, 398, 190, "papa")
    persona(c, 288, 398, 172, "mama")
    persona(c, 348, 398, 88, "nino")
    return dict(man=(455, 209 - 13.5, 24))


def foto_2004(c):
    c.vgrad(0, 0, PW, 70, (176, 204, 228), (214, 224, 230))
    casa(c, PAL_DAY)
    c.rect(462, 120, 530, 336, f=(156, 142, 118))
    man(c, 486, 336, 108)
    persona(c, 110, 400, 150, "abuela")
    persona(c, 180, 402, 188, "papa")
    persona(c, 250, 402, 168, "mama")
    persona(c, 318, 402, 108, "nino")
    return dict(man=(486, 336 - 54, 58))


def foto_2011(c):
    c.vgrad(0, 0, PW, 70, (150, 184, 220), (200, 214, 226))
    casa(c, PAL_DAY, puerta_abierta=True)
    man(c, 428, 336, 142, lum=1.0)
    persona(c, 200, 402, 186, "papa")
    persona(c, 268, 402, 166, "mama")
    persona(c, 330, 402, 128, "nino")
    persona(c, 420, 376, 150, "abuela")
    return dict(man=(428, 336 - 71, 74))


def foto_2023(c):
    c.vgrad(0, 0, PW, 70, (18, 22, 50), (60, 54, 90))
    casa(c, PAL_DUSK)
    x, y, w, h = WIN
    c.rect(x - 5, y - 5, x + w + 5, y + h + 5, f=PAL_DUSK["trim"])
    c.rect(x, y, x + w, y + h, f=(8, 10, 20))
    man_bust(c, x + w / 2, y + 36, 16.5, lum=1.45)
    c.poly([(x + 6, y), (x + 30, y), (x + 8, y + h)], (30, 36, 60))        # reflejo del vidrio
    c.line([(x + w / 2, y), (x + w / 2, y + h)], PAL_DUSK["trim"], 1.2)    # un marco fino
    c.rect(x - 9, y + h + 5, x + w + 9, y + h + 11, f=PAL_DUSK["trim"])
    persona(c, 120, 404, 146, "abuela")
    persona(c, 194, 406, 190, "papa")
    persona(c, 268, 406, 170, "mama")
    persona(c, 336, 406, 158, "nino")
    return dict(man=(x + w / 2, y + 40, 40), win=(x + w / 2, y + h / 2), empty=(486, 336 - 54, 58))


# =====================================================================
# 4. FOTOS ANTIGUAS: revelado, grano, rayones, marco y sombra
# =====================================================================
PW, PH = 560, 420


def revelar(img, sat=1.0, tint=(1, 1, 1), fade_=0.0, vig=0.3, grain=5, blur=0.6, flash=None, seed=1):
    rng = np.random.default_rng(seed)
    a = np.asarray(img, np.float32) / 255.0
    g = a.mean(2, keepdims=True)
    a = g + (a - g) * sat
    a = a * np.array(tint, np.float32)[None, None, :]
    h, w = a.shape[:2]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    if flash is not None:
        fx, fy, fs, lo, hi = flash
        m = np.exp(-(((x - fx) ** 2) + ((y - fy) ** 2)) / (fs * fs))
        a = a * (lo + (hi - lo) * m)[..., None]
    a = a * (1 - vig * np.clip(x * x + y * y - 0.25, 0, None))[..., None]
    a = a * (1 - fade_) + fade_ * np.array([.55, .50, .42], np.float32)
    a = a + rng.normal(0, grain / 255.0, (h, w, 1)).astype(np.float32)
    out = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else out


def enmarcar(img, k, seed=1):
    rng = np.random.default_rng(seed)
    b = int(16 * k)
    pad = int(46 * k)
    w, h = img.size
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for _ in range(4):                                        # rayones finos
        x = int(rng.integers(0, w)); y0 = int(rng.integers(0, h // 2)); y1 = y0 + int(rng.integers(h // 8, h // 2))
        d.line([(x, y0), (x + int(rng.integers(-6, 6)), y1)], fill=(240, 236, 220, 55), width=max(1, int(k * 0.5)))
    for _ in range(40):                                       # polvo
        x = int(rng.integers(0, w)); y = int(rng.integers(0, h)); r = int(rng.integers(1, int(k * 1.6) + 2))
        d.ellipse([x, y, x + r, y + r], fill=(235, 230, 215, 70))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    paper = Image.new("RGB", (w + 2 * b, h + 2 * b), (228, 220, 198))
    paper.paste(img, (b, b))
    L = Image.new("RGBA", (paper.width + 2 * pad, paper.height + 2 * pad), (0, 0, 0, 0))
    sh = Image.new("L", L.size, 0)
    ImageDraw.Draw(sh).rectangle([pad + 7 * k, pad + 12 * k, pad + paper.width + 7 * k, pad + paper.height + 12 * k], fill=175)
    sh = sh.filter(ImageFilter.GaussianBlur(14 * k))
    L.paste((0, 0, 0, 255), (0, 0), sh)
    L.paste(paper, (pad, pad))
    return L


def hacer_foto(year):
    t0 = time.time()
    final = (year == 2023)
    k, ss = (4, 1.5) if final else (3, 2.0)
    c = Cv(PW, PH, k * ss)
    fn = {1998: foto_1998, 2004: foto_2004, 2011: foto_2011, 2023: foto_2023}[year]
    meta = fn(c)
    img = c.im.resize((int(PW * k), int(PH * k)), LANCZOS)
    del c
    if year == 1998:
        img = revelar(img, sat=.78, tint=(1.06, 1.0, .88), fade_=.20, vig=.35, grain=7, blur=.9, seed=1)
    elif year == 2004:
        img = revelar(img, sat=.9, tint=(1.04, 1.0, .93), fade_=.12, vig=.3, grain=6, blur=.7, seed=2)
    elif year == 2011:
        img = revelar(img, sat=1.0, tint=(1.0, 1.0, .98), fade_=.06, vig=.3, grain=5, blur=.6, seed=3)
    else:
        img = revelar(img, sat=1.05, tint=(.98, .99, 1.04), fade_=.03, vig=.5, grain=5, blur=.5,
                      flash=(-.18, .55, .85, .62, 1.18), seed=4)
    L = enmarcar(img, k, seed=year)
    levels = [(float(k), L)]
    for f in (2, 4):
        levels.append((k / f, L.resize((L.width // f, L.height // f), LANCZOS)))
    print("   foto %d lista (%.1fs)" % (year, time.time() - t0))
    return levels, meta


# =====================================================================
# 5. LA MESA (fondo) y las luces
# =====================================================================
BGW, BGH, BGOX, BGOY = 1520, 3800, 760, 1100


def hacer_mesa():
    rng = np.random.default_rng(3)
    img = np.zeros((BGH, BGW, 3), np.float32)
    pw = 190
    cols = []
    for i in range(BGW // pw + 2):
        x0 = i * pw
        col = np.array([76, 47, 30], np.float32) * (0.82 + 0.3 * rng.random()) * np.array([1, .98 + .05 * rng.random(), .95 + .1 * rng.random()])
        img[:, x0:x0 + pw] = col
        cols.append(x0)

    def ruido(hh, ww):
        n = (rng.random((hh, ww)) * 255).astype(np.uint8)
        return np.asarray(Image.fromarray(n).resize((BGW, BGH), BICUBIC), np.float32) / 255.0

    g = 0.78 + 0.42 * ruido(BGH // 70, BGW // 5) + 0.2 * ruido(BGH // 12, BGW // 3)
    img *= g[..., None]
    for x0 in cols:
        img[:, x0:x0 + 4] *= 0.3
        for _ in range(3):
            y = int(rng.integers(0, BGH - 4))
            img[y:y + 4, x0:x0 + pw] *= 0.35
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)

    def P(wx, wy):
        return (wx + BGOX, wy + BGOY)

    cx, cy = P(-430, 330)                                            # taza de café
    d.ellipse([cx - 78, cy - 78, cx + 78, cy + 78], fill=(222, 218, 208))
    d.ellipse([cx - 52, cy - 52, cx + 52, cy + 52], fill=(236, 232, 224), outline=(190, 184, 172), width=3)
    d.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(52, 30, 20))
    d.arc([cx + 40, cy - 22, cx + 92, cy + 22], -90, 90, fill=(236, 232, 224), width=9)
    ax, ay = P(450, 760)                                             # álbum viejo
    d.polygon([(ax - 120, ay - 150), (ax + 120, ay - 140), (ax + 126, ay + 150), (ax - 116, ay + 156)], fill=(92, 28, 30))
    d.polygon([(ax - 100, ay - 128), (ax + 100, ay - 120), (ax + 104, ay + 128), (ax - 96, ay + 134)], outline=(150, 110, 70), width=4)
    px, py = P(-420, 930)                                            # lápiz
    d.polygon([(px - 120, py - 8), (px + 90, py - 14), (px + 130, py), (px + 90, py + 14), (px - 120, py + 8)], fill=(210, 170, 60))
    d.polygon([(px + 90, py - 14), (px + 130, py), (px + 90, py + 14)], fill=(220, 190, 150))
    kx, ky = P(430, 1320)                                            # llave
    d.ellipse([kx - 34, ky - 34, kx + 34, ky + 34], outline=(170, 150, 90), width=9)
    d.rectangle([kx + 30, ky - 6, kx + 150, ky + 6], fill=(170, 150, 90))
    d.rectangle([kx + 120, ky + 6, kx + 132, ky + 28], fill=(170, 150, 90))
    return im.filter(ImageFilter.GaussianBlur(2.4))


def hacer_luz():
    x = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    lamp = np.exp(-(((x + 0.15) ** 2) / 0.9 + ((y + 0.25) ** 2) / 0.8))
    base = (0.45 + 0.75 * lamp)
    vig = 1 - 0.50 * np.clip(x * x * 0.8 + y * y * 0.55, 0, 1.3)
    tint = np.array([1.06, 0.98, 0.88], np.float32)[None, None, :]
    normal = (base * vig)[..., None] * tint
    dark = (np.power(np.clip(base * vig, 0.01, 2), 1.7) * 0.8)[..., None] * np.array([0.92, 0.96, 1.08], np.float32)[None, None, :]
    return normal.astype(np.float32), dark.astype(np.float32)


# =====================================================================
# 6. CÁMARA
# =====================================================================
def mat_T(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], float)


def mat_S(s):
    return np.diag([s, s, 1.0])


def mat_R(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


class Cam:
    def __init__(s, keys):
        keys = sorted(keys, key=lambda k: k[0])
        fixed = []
        for kk in keys:
            if fixed and kk[0] <= fixed[-1][0] + 0.04:
                kk = (fixed[-1][0] + 0.04,) + tuple(kk[1:])
            fixed.append(kk)
        s.t = np.array([k[0] for k in fixed], float)
        s.v = np.array([[k[1], k[2], math.log(k[3]), k[4]] for k in fixed], float)
        n = len(fixed)
        m = np.zeros_like(s.v)
        for i in range(1, n - 1):
            same = np.allclose(s.v[i], s.v[i - 1], atol=1e-6) or np.allclose(s.v[i], s.v[i + 1], atol=1e-6)
            if not same:
                m[i] = (s.v[i + 1] - s.v[i - 1]) / (s.t[i + 1] - s.t[i - 1])
        s.m = m

    def at(s, t):
        if t <= s.t[0]:
            v = s.v[0]
        elif t >= s.t[-1]:
            v = s.v[-1]
        else:
            i = int(np.searchsorted(s.t, t)) - 1
            h = s.t[i + 1] - s.t[i]
            u = (t - s.t[i]) / h
            h00 = 2 * u ** 3 - 3 * u ** 2 + 1
            h10 = u ** 3 - 2 * u ** 2 + u
            h01 = -2 * u ** 3 + 3 * u ** 2
            h11 = u ** 3 - u ** 2
            v = h00 * s.v[i] + h10 * h * s.m[i] + h01 * s.v[i + 1] + h11 * h * s.m[i + 1]
        return float(v[0]), float(v[1]), float(math.exp(v[2])), float(v[3])


LAYOUT = [dict(year=1998, pos=(-25, 0), rot=-2.5), dict(year=2004, pos=(30, 520), rot=2.0),
          dict(year=2011, pos=(-15, 1040), rot=-1.5), dict(year=2023, pos=(10, 1560), rot=1.0)]


def wp(i, lx, ly):
    a = math.radians(LAYOUT[i]["rot"])
    dx, dy = lx - PW / 2, ly - PH / 2
    return (LAYOUT[i]["pos"][0] + dx * math.cos(a) - dy * math.sin(a),
            LAYOUT[i]["pos"][1] + dx * math.sin(a) + dy * math.cos(a))


def screen2world(cam):
    cx, cy, z, rot = cam
    return mat_T(cx, cy) @ mat_R(rot) @ mat_S(1 / z) @ mat_T(-W / 2, -H / 2)


def w2s(cam, x, y):
    cx, cy, z, rot = cam
    a = math.radians(-rot)
    dx, dy = x - cx, y - cy
    rx = dx * math.cos(a) - dy * math.sin(a)
    ry = dx * math.sin(a) + dy * math.cos(a)
    return rx * z + W / 2, ry * z + H / 2


def transformar(img, M, size=(W, H)):
    return img.transform(size, AFFINE, (M[0, 0], M[0, 1], M[0, 2], M[1, 0], M[1, 1], M[1, 2]), resample=BICUBIC)


def pintar_capa(canvas, capa, cam):
    cx, cy, z, rot = cam
    px, py = LAYOUT[capa["i"]]["pos"]
    sx, sy = w2s(cam, px, py)
    if math.hypot(sx - W / 2, sy - H / 2) > (math.hypot(PW, PH) / 2 + 40) * z + math.hypot(W, H) / 2:
        return canvas
    lv = capa["levels"]
    chosen = lv[0]
    for kk, im in lv:
        if kk >= z:
            chosen = (kk, im)
    kk, im = chosen
    M = (mat_T(im.width / 2, im.height / 2) @ mat_S(kk) @ mat_R(-LAYOUT[capa["i"]]["rot"])
         @ mat_T(-px, -py) @ screen2world(cam))
    layer = transformar(im, M)
    return Image.alpha_composite(canvas, layer)


# =====================================================================
# 7. CAPA DE GRÁFICOS (anillos neón, líneas, textos) con brillo
# =====================================================================
GS = 3


class Ov:
    def __init__(s):
        s.im = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
        s.d = ImageDraw.Draw(s.im)
        s.gl = Image.new("RGB", (W // GS, H // GS), (0, 0, 0))
        s.g = ImageDraw.Draw(s.gl)
        s.used = s.gused = False

    def arc(s, x, y, r, p, col, wd=3.0, a=1.0, glow=1.0):
        if a <= 0.01 or p <= 0.001:
            return
        s.used = True
        box = [(x - r) * 2, (y - r) * 2, (x + r) * 2, (y + r) * 2]
        c = tuple(col) + (int(255 * a),)
        if p >= 0.999:
            s.d.ellipse(box, outline=c, width=int(wd * 2))
        else:
            s.d.arc(box, -90, -90 + 360 * p, fill=c, width=int(wd * 2))
        if glow > 0:
            s.gused = True
            gb = [(x - r) / GS, (y - r) / GS, (x + r) / GS, (y + r) / GS]
            gc = tuple(int(v * a * glow) for v in col)
            if p >= 0.999:
                s.g.ellipse(gb, outline=gc, width=max(1, int(wd * 2.2 / GS)))
            else:
                s.g.arc(gb, -90, -90 + 360 * p, fill=gc, width=max(1, int(wd * 2.2 / GS)))

    def poly(s, pts, col, wd=2.5, a=1.0, glow=1.0):
        if a <= 0.01 or len(pts) < 2:
            return
        s.used = True
        s.d.line([(x * 2, y * 2) for x, y in pts], fill=tuple(col) + (int(255 * a),), width=int(wd * 2))
        if glow > 0:
            s.gused = True
            s.g.line([(x / GS, y / GS) for x, y in pts], fill=tuple(int(v * a * glow) for v in col), width=max(1, int(wd * 2 / GS)))

    def dot(s, x, y, r, col, a=1.0, glow=1.0):
        if a <= 0.01:
            return
        s.used = True
        s.d.ellipse([(x - r) * 2, (y - r) * 2, (x + r) * 2, (y + r) * 2], fill=tuple(col) + (int(255 * a),))
        if glow > 0:
            s.gused = True
            s.g.ellipse([(x - r * 2) / GS, (y - r * 2) / GS, (x + r * 2) / GS, (y + r * 2) / GS], fill=tuple(int(v * a * glow * .6) for v in col))

    def text(s, txt, x, y, size, col, a=1.0, track=0.0, mono=False, bold=False, glow=0.0):
        if a <= 0.01:
            return
        s.used = True
        f = F(size * 2, mono, bold)
        ws = [s.d.textlength(ch, font=f) for ch in txt]
        tot = sum(ws) + track * 2 * (len(txt) - 1)
        xx = x * 2 - tot / 2
        yy = y * 2
        fg = F(max(6, size / GS), mono, bold)
        gx = xx
        for ch, wch in zip(txt, ws):
            s.d.text((xx + 2, yy + 3), ch, font=f, fill=(0, 0, 0, int(190 * a)), anchor="lm")
            s.d.text((xx, yy), ch, font=f, fill=tuple(col) + (int(255 * a),), anchor="lm")
            if glow > 0:
                s.gused = True
                s.g.text((gx / 2 / GS, yy / 2 / GS), ch, font=fg, fill=tuple(int(v * a * glow * .8) for v in col), anchor="lm")
            xx += wch + track * 2
            gx += wch + track * 2

    def compose(s, canvas):
        if s.gused:
            g = s.gl.filter(ImageFilter.GaussianBlur(1.7)).resize((W, H), BILINEAR)
            g = ImageChops.add(g, g)
            canvas = ImageChops.add(canvas, g)
        if s.used:
            o = s.im.reduce(2)
            canvas = Image.alpha_composite(canvas.convert("RGBA"), o).convert("RGB")
        return canvas


# =====================================================================
# 8. VOZ (edge-tts, gratis) y cronograma
# =====================================================================
def _estimar(texto):
    return len(texto.split()) * 0.36 + 0.25 + 0.3 * texto.count("…")


def sintetizar_linea(clave, texto, rate, voz):
    mp3 = os.path.join(WORK, "v_%s_%d_%s.mp3" % (clave, rate, voz))
    if os.path.exists(mp3) and os.path.getsize(mp3) > 2000:
        return mp3
    cmd = [sys.executable, "-m", "edge_tts", "--voice", voz, "--rate=%+d%%" % rate, "--pitch=" + TONO,
           "--text", texto, "--write-media", mp3]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        pass
    return mp3 if (os.path.exists(mp3) and os.path.getsize(mp3) > 2000) else None


def cargar_voz(mp3):
    wav = mp3[:-4] + ".wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", mp3, "-ar", str(SR), "-ac", "1", wav], check=True)
    w = wave.open(wav)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    w.close()
    idx = np.where(np.abs(x) > 0.012)[0]
    if len(idx) == 0:
        return x
    return x[max(0, idx[0] - int(.03 * SR)): min(len(x), idx[-1] + int(.10 * SR))]


def cronograma(durs):
    ST, EN = {}, {}
    t = 0.30
    for (k, _, pausa), d in zip(LINEAS, durs):
        t += pausa
        ST[k] = t
        EN[k] = t + d
        t = EN[k]
    t_cut = EN["s5c"] + 1.15
    return ST, EN, t_cut, t_cut + 0.6


def preparar_voz():
    rate = VELOCIDAD_BASE
    voces = [VOZ] + VOCES_ALTERNAS
    for intento in range(3):
        clips, usada = None, None
        for voz in voces:
            cl = []
            for k, txt, _ in LINEAS:
                m = sintetizar_linea(k, txt, rate, voz)
                if m is None:
                    cl = None
                    break
                cl.append(cargar_voz(m))
            if cl is not None:
                clips, usada = cl, voz
                break
        if clips is None:
            print("   ⚠ NO se pudo generar la voz (¿sin internet o edge-tts no instalado?).")
            print("   ⚠ El video saldrá SIN narración, solo con música y efectos.")
            durs = [_estimar(t) for _, t, _ in LINEAS]
            ST, EN, t_cut, T = cronograma(durs)
            return None, ST, EN, t_cut, T
        durs = [len(c) / SR for c in clips]
        ST, EN, t_cut, T = cronograma(durs)
        print("   voz %s, velocidad %+d%% -> el video dura %.1f s" % (usada, rate, T))
        lo, hi = DURACION_OBJETIVO
        if lo <= T <= hi:
            break
        rate = int(clamp(rate + (6 if T > hi else -6) * (1 if abs(T - (lo + hi) / 2) < 8 else 2), -15, 28))
    voice = np.zeros(int(T * SR) + SR, np.float32)
    for (k, _, _), c in zip(LINEAS, clips):
        a = int(ST[k] * SR)
        voice[a:a + len(c)] += c
    voice = voice[:int(T * SR)]
    pk = np.max(np.abs(voice)) + 1e-9
    return voice / pk * 0.85, ST, EN, t_cut, T


# =====================================================================
# 9. SONIDO (música ambiental y efectos, generados por código)
# =====================================================================
def _t(n):
    return np.arange(n, dtype=np.float32) / SR


def _noise(n, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def _band(x, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x)).astype(np.float32)


def _nrm(x, p=1.0):
    return x / (np.max(np.abs(x)) + 1e-9) * p


def fx(name):
    if name == "shutter":
        n = int(.35 * SR)
        x = np.zeros(n, np.float32)
        a = _band(_noise(int(.014 * SR), 1), 1200, 9000) * np.hanning(int(.014 * SR))
        x[:len(a)] += a * 1.0
        b = _band(_noise(int(.035 * SR), 2), 500, 6000) * np.hanning(int(.035 * SR))
        x[int(.075 * SR):int(.075 * SR) + len(b)] += b * .8
        t = _t(n)
        x += np.sin(2 * np.pi * 150 * t) * np.exp(-t * 40) * .4
        return _nrm(x, .9)
    if name == "paper":
        n = int(.75 * SR)
        t = _t(n)
        env = np.abs(_band(_noise(n, 5), 3, 28)) ** 1.5
        env = env / (env.max() + 1e-9) * np.hanning(n)
        x = _band(_noise(n, 6), 1800, 9500) * env
        return _nrm(x, .6)
    if name == "whoosh":
        n = int(1.3 * SR)
        t = _t(n)
        x = _band(_noise(n, 7), 250, 3200) * np.abs(np.sin(np.pi * t / t[-1])) ** 2.2
        return _nrm(x, .7)
    if name == "ping":
        n = int(.6 * SR)
        t = _t(n)
        x = (np.sin(2 * np.pi * 1318 * t) + .35 * np.sin(2 * np.pi * 1975 * t)) * np.exp(-t * 7)
        return _nrm(x, .45)
    if name == "beat":
        n = int(.6 * SR)
        t = _t(n)
        x = np.sin(2 * np.pi * 52 * t * (1 - .35 * t)) * np.exp(-t * 15)
        t2 = np.clip(t - .22, 0, None)
        x2 = np.sin(2 * np.pi * 46 * t2) * np.exp(-t2 * 17) * (t > .22) * .7
        return _nrm(x + x2, .95)
    if name == "creak":
        n = int(1.2 * SR)
        t = _t(n)
        f = 62 + 30 * np.sin(2 * np.pi * .9 * t) + 18 * t
        ph = np.cumsum(2 * np.pi * f / SR)
        saw = 2 * ((ph / (2 * np.pi)) % 1) - 1
        x = _band(saw, 70, 1100) * (1 + .7 * np.sin(2 * np.pi * 21 * t)) * np.abs(np.sin(np.pi * t / t[-1])) ** 1.3
        return _nrm(x, .6)
    if name == "boom":
        n = int(2.4 * SR)
        t = _t(n)
        x = np.sin(2 * np.pi * 40 * t * (1 - .1 * t)) * np.exp(-t * 2.0) + _band(_noise(n, 9), 25, 220) * np.exp(-t * 4) * .9
        return _nrm(x, 1.0)
    if name == "riser":
        n = int(1.4 * SR)
        t = _t(n)
        r = (t / t[-1]) ** 2.3
        ph = np.cumsum(2 * np.pi * (180 + 1500 * (t / t[-1]) ** 2) / SR)
        x = _band(_noise(n, 10), 500, 7000) * r * .8 + np.sin(ph) * r * .35
        return _nrm(x, .7)
    if name == "cricket":
        n = int(.12 * SR)
        t = _t(n)
        env = np.zeros(n, np.float32)
        for k in range(3):
            a = int(k * .036 * SR)
            L = int(.018 * SR)
            env[a:a + L] = np.hanning(L)
        return _nrm(np.sin(2 * np.pi * 4300 * t) * env, .5)
    raise ValueError(name)


def poner(track, sig, t, g=1.0):
    a = int(t * SR)
    if a >= len(track) or a < 0:
        return
    b = min(len(track), a + len(sig))
    track[a:b] += sig[:b - a] * g


def hacer_musica(T, ST, EN, t_rev, t_cut):
    n = int(T * SR)
    t = _t(n)
    tension = np.clip((t - ST["s3a"]) / (ST["s5a"] - ST["s3a"]), 0, 1) ** 1.2
    m = np.zeros(n, np.float32)
    for i, (f, a) in enumerate([(55.0, .50), (82.4, .28), (110.0, .20), (130.8, .15), (164.8, .06)]):
        m += a * np.sin(2 * np.pi * f * t + .3 * np.sin(2 * np.pi * (.05 + .013 * i) * t + i))
    m *= .75 + .25 * np.sin(2 * np.pi * .09 * t)
    air = _band(_noise(n, 12), 150, 1400) * (.07 + .04 * np.sin(2 * np.pi * .13 * t))
    glass = (np.sin(2 * np.pi * 880 * t) + .8 * np.sin(2 * np.pi * 932.3 * t)) * (1 + .3 * np.sin(2 * np.pi * 4.7 * t))
    m = m + air + glass * .05 * tension
    m *= np.clip(t / 1.2, 0, 1)
    return m


def hacer_audio(voice, ST, EN, T, t_rev, t_cut, sfx_ev):
    n = int(T * SR)
    music = hacer_musica(T, ST, EN, t_rev, t_cut)
    v = voice if voice is not None else np.zeros(n, np.float32)
    env = np.convolve(np.abs(v), np.ones(int(.14 * SR)) / int(.14 * SR), mode="same")
    act = np.clip(env * 9, 0, 1)
    g = 0.62 * (1 - 0.55 * act)
    t = _t(n)
    sil0, sil1 = EN["s5a"] + .05, t_rev - 1.25
    s = np.clip((t - sil0) / .5, 0, 1) * np.clip((sil1 - t) / .01, 0, 1)
    g = g * (1 - .94 * s)
    back = np.clip((t - (t_rev + .1)) / 1.2, 0, 1)
    g = g + .35 * back * np.clip((t_cut - t) / .05, 0, 1)
    out = music * g + v
    fxs = np.zeros(n, np.float32)
    for (tt, name, gain) in sfx_ev:
        poner(fxs, fx(name), tt, gain)
    out = out + fxs
    out[int(t_cut * SR):] = 0
    k = int(.03 * SR)
    a0 = int(t_cut * SR)
    out[max(0, a0 - k):a0] *= np.linspace(1, 0, min(k, a0))
    out = out / (np.max(np.abs(out)) + 1e-9) * .9
    path = os.path.join(WORK, "mezcla.wav")
    w = wave.open(path, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((out * 32767).astype(np.int16).tobytes())
    w.close()
    return path


# =====================================================================
# 10. GUION VISUAL: cámara, anillos, textos, efectos (todo sincronizado con la voz)
# =====================================================================
CIAN, VIOLETA, ROJO = (0, 229, 255), (170, 112, 255), (255, 52, 72)


def construir_guion(ST, EN, M, t_cut):
    S = lambda k, o=0.0: ST[k] + o
    E = lambda k, o=0.0: EN[k] + o
    m = [wp(i, M[i]["man"][0], M[i]["man"][1]) for i in range(3)]
    win = wp(3, *M[3]["win"])
    empty = wp(3, *M[3]["empty"][:2])
    t_rev = E("s5b", -1.0)
    K = lambda t, p, z, r=0.0: (t, p[0], p[1], z, r)
    keys = [
        K(0.0, wp(0, PW / 2 + 10, PH / 2 + 5), 1.16, -0.8),
        K(S("s1b", -0.1), wp(0, PW / 2 + 40, PH / 2), 1.35, -0.4),
        K(E("s1b", 0.2), (m[0][0] - 4, m[0][1] + 4), 3.7, 0),
        K(S("s2a", 0.9), (m[0][0] - 4, m[0][1] + 4), 3.7, 0),
        K(S("s2a", 3.8), (0, 570), 0.70, 0.6),
        K(E("s2b", 0.1), (0, 610), 0.72, 0.2),
        K(S("s3b", 0.6), (m[0][0] - 30, m[0][1] - 5), 2.1, 0),
        K(E("s3b", 0.3), (m[0][0] - 30, m[0][1] - 5), 2.1, 0),
        K(S("s3c", 1.0), (m[1][0] - 35, m[1][1]), 1.6, 0),
        K(E("s3c", 0.3), (m[1][0] - 35, m[1][1]), 1.6, 0),
        K(S("s4a", 1.1), (m[2][0] - 30, m[2][1] + 10), 1.7, 0),
        K(E("s4a", 0.5), (m[2][0] - 30, m[2][1] + 10), 1.7, 0),
        K(S("s4b", 1.7), (0, 545), 0.70, 0.4),
        K(E("s4b", 0.2), (0, 565), 0.72, 0.2),
        K(E("s4c", 0.3), wp(3, PW / 2, PH / 2), 1.08, 0.3),
        K(S("s5a", 1.1), wp(3, PW / 2 + 60, PH / 2 + 40), 1.25, 0.6),
        K(E("s5a", 0.2), wp(3, PW / 2 + 105, PH / 2 + 30), 1.32, 0.2),
        K(S("s5b", 1.3), wp(3, PW / 2 + 70, PH / 2 - 40), 1.7, 0),
        K(t_rev - 0.4, (win[0] - 8, win[1] + 6), 3.4, 0),
        K(t_rev + 0.18, (win[0], win[1] + 2), 6.4, 0),
        K(t_cut, (win[0], win[1] + 2), 6.9, 0),
    ]
    ev = []

    def ring(t0, t1, pt, r, col, draw=.6, out=.45, flick=None):
        ev.append(dict(k="ring", t0=t0, t1=t1, pt=pt, r=r, col=col, draw=draw, out=out, flick=flick))

    def txt(t0, t1, s, y, size, col, track=3, mono=False, bold=False, glow=0.0, amax=1.0):
        ev.append(dict(k="txt", t0=t0, t1=t1, s=s, y=y, size=size, col=col, track=track, mono=mono, bold=bold, glow=glow, amax=amax))

    if ETIQUETA_FICCION:
        txt(0.5, 3.6, ETIQUETA_FICCION, 178, 21, (205, 205, 212), track=5, mono=True, amax=.75)
    ring(S("s1b", .45), S("s2a", .9), m[0], 24, CIAN, .7)
    ring(S("s2a", 2.0), S("s3a", .5), m[0], 46, CIAN)
    ring(S("s2b", .15), S("s3a", .5), m[1], 62, CIAN)
    ring(S("s2b", 1.0), S("s3a", .5), m[2], 78, CIAN)
    txt(S("s2a", 2.3), S("s2a", 4.9), "nadie lo recordaba", 245, 38, (236, 236, 240), track=4)
    ring(S("s3b", .45), E("s3b", .35), m[0], 24, CIAN)
    txt(S("s3b", .5), E("s3b", .45), "1998", 232, 70, CIAN, track=8, mono=True, bold=True, glow=1.0)
    ring(S("s3c", .9), E("s3c", .35), m[1], 62, CIAN)
    txt(S("s3c", .3), E("s3c", .5), "2004", 232, 70, CIAN, track=8, mono=True, bold=True, glow=1.0)
    ring(S("s4a", .3), E("s4a", .6), m[2], 80, VIOLETA)
    txt(S("s4a", .0), E("s4a", .5), "2011", 232, 70, VIOLETA, track=8, mono=True, bold=True, glow=1.0)
    ev.append(dict(k="line", t0=S("s4b", .35), t1=S("s4c", .75), pts=m, draw=2.3, col=CIAN))
    txt(S("s4b", 1.1), E("s4b", .9), "cada vez más cerca", 250, 38, (236, 236, 240), track=4)
    ring(S("s5a", .9), E("s5a", .4), empty, 64, CIAN, .5, .25, flick=S("s5a", 1.9))
    ring(t_rev - .12, t_cut, win, 41, ROJO, .35, .05)
    txt(t_rev + 1.0, t_cut - .05, "¿QUIÉN ES?", 925, 40, ROJO, track=9, bold=True, glow=1.0)

    sfx = [(0.0, "shutter", .9), (0.05, "paper", .35)]
    sfx += [(e["t0"], "ping", .22) for e in ev if e["k"] == "ring" and e["col"] != ROJO and e["t0"] > 1]
    sfx += [(S("s2a", .5), "paper", .3), (S("s2a", 1.4), "whoosh", .22), (S("s3a", 1.1), "whoosh", .28),
            (S("s3c", .5), "paper", .3), (S("s4a", -.1), "whoosh", .25), (S("s4b", 1.2), "whoosh", .22),
            (S("s4c", 1.3), "whoosh", .45)]
    tb = S("s4b", 0.2)
    k = 0
    while tb < EN["s5a"] - .2:
        sfx.append((tb, "beat", .22 + .30 * clamp((tb - S("s4b")) / (E("s5a") - S("s4b")))))
        tb += 1.05 - .35 * clamp((tb - S("s4b")) / (E("s5a") - S("s4b")))
    tc = S("s5a", .1)
    while tc < t_rev - 1.2:
        sfx.append((tc, "cricket", .09 + .02 * (k % 3)))
        tc += .30 + .11 * ((k * 7) % 4)
        k += 1
    sfx += [(t_rev - 1.3, "riser", .55), (t_rev - .02, "boom", .95), (t_rev + .02, "creak", .45), (t_rev, "ping", .0)]

    flick = [(S("s5a", 1.35), .14, .5), (E("s5a", .6), .2, .42), (t_rev - .7, .1, .5), (t_rev - .35, .08, .55)]
    return dict(cam=Cam(keys), ev=ev, sfx=sfx, flick=flick, t_rev=t_rev, t_cut=t_cut, t_s5=S("s5a"),
                dark=[(0, 0.0), (S("s4b"), 0.0), (E("s4c"), .35), (S("s5b"), .6), (t_rev, .75), (t_cut, .75)])


# =====================================================================
# 11. RENDER DE CADA FOTOGRAMA
# =====================================================================
CTX = {}


def render_frame(i):
    C = CTX
    t = i / FPS
    if t >= C["t_cut"]:
        return C["black"]
    cam = list(C["cam"].at(t))
    amp = 0.45 + 2.2 * smooth((t - C["t_s5"]) / max(0.5, C["t_rev"] - C["t_s5"]))
    cam[0] += amp * (math.sin(t * 2.3) + .6 * math.sin(t * 5.7 + 1)) / cam[2]
    cam[1] += amp * (math.sin(t * 1.9 + 2) + .5 * math.sin(t * 6.1)) / cam[2]
    cam[3] += .3 * math.sin(t * 1.3)
    zb = cam[2] * 0.9
    Mb = mat_T(BGOX, BGOY) @ mat_T(cam[0] * .9, cam[1] * .9) @ mat_R(cam[3]) @ mat_S(1 / zb) @ mat_T(-W / 2, -H / 2)
    canvas = transformar(C["bg"], Mb).convert("RGBA")
    for capa in C["layers"]:
        canvas = pintar_capa(canvas, capa, cam)
    canvas = canvas.convert("RGB")
    if t < 0.75:
        canvas = canvas.filter(ImageFilter.GaussianBlur(7 * (1 - t / 0.75) ** 2))

    dk = float(np.interp(t, [d[0] for d in C["dark"]], [d[1] for d in C["dark"]]))
    L = C["luz_n"] * (1 - dk) + C["luz_d"] * dk
    b = 1 + .02 * math.sin(t * 9.1) + .015 * math.sin(t * 23.0)
    for (t0, dur, depth) in C["flick"]:
        if t0 <= t < t0 + dur and i % 2 == 0:
            b *= 1 - depth
    arr = np.asarray(canvas, np.float32) * L * b
    dt = t - C["t_rev"]
    if 0 <= dt < 0.16:
        arr += np.array([70, 6, 10], np.float32) * (1 - dt / .16)
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    ov = Ov()
    for (bx, by, d, sz) in C["motes"]:
        x = (bx + 9 * d * t - .22 * d * cam[0] * cam[2]) % W
        y = (by - 7 * d * t - .22 * d * cam[1] * cam[2]) % H
        ov.dot(x, y, sz, (255, 232, 190), a=.35 + .25 * math.sin(t * 2 + bx), glow=.3)
    for e in C["ev"]:
        if t < e["t0"] or t > e["t1"]:
            continue
        if e["k"] == "ring":
            p = smooth((t - e["t0"]) / e["draw"])
            al = clamp((t - e["t0"]) / .15) * clamp((e["t1"] - t) / e["out"])
            if e["flick"] and t > e["flick"]:
                al *= (0.25 + 0.75 * (math.sin(t * 61) > 0.1)) * clamp(1 - (t - e["flick"]) / max(0.2, e["t1"] - e["flick"]))
            pop = 1 + .4 * math.exp(-(t - e["t0"]) * 7)
            x, y = w2s(cam, *e["pt"])
            if -300 < x < W + 300 and -300 < y < H + 300:
                ov.arc(x, y, e["r"] * cam[2] * pop * (1 + .03 * math.sin(t * 6)), p, e["col"], wd=3.2, a=al)
        elif e["k"] == "line":
            pts = [w2s(cam, *q) for q in e["pts"]]
            seg = [math.hypot(pts[j + 1][0] - pts[j][0], pts[j + 1][1] - pts[j][1]) for j in range(len(pts) - 1)]
            tot = sum(seg)
            lim = tot * smooth((t - e["t0"]) / e["draw"])
            al = clamp((e["t1"] - t) / .5)
            pos = 0.0
            for j, l in enumerate(seg):
                s0 = 0.0
                while s0 < l:
                    a0, a1 = pos + s0, min(pos + s0 + 16, pos + l)
                    if a0 >= lim:
                        break
                    a1 = min(a1, lim)
                    f0, f1 = (a0 - pos) / l, (a1 - pos) / l
                    P0 = (pts[j][0] + (pts[j + 1][0] - pts[j][0]) * f0, pts[j][1] + (pts[j + 1][1] - pts[j][1]) * f0)
                    P1 = (pts[j][0] + (pts[j + 1][0] - pts[j][0]) * f1, pts[j][1] + (pts[j + 1][1] - pts[j][1]) * f1)
                    ov.poly([P0, P1], e["col"], 2.6, al)
                    s0 += 26
                pos += l
                if lim >= pos - 1e-3:
                    ov.dot(pts[j + 1][0], pts[j + 1][1], 5.5, e["col"], al)
            ov.dot(pts[0][0], pts[0][1], 5.5, e["col"], al)
        else:
            a = fade(t, e["t0"], e["t1"], .45, .45) * e["amax"]
            slide = (1 - smooth((t - e["t0"]) / .5)) * 14
            ov.text(e["s"], W / 2, e["y"] + slide, e["size"], e["col"], a, e["track"], e["mono"], e["bold"], e["glow"])
    canvas = ov.compose(canvas)

    a = np.asarray(canvas, np.int16) + C["noise"][i % 6]
    if 0.03 < dt < 0.30:
        rng = np.random.default_rng(i)
        for _ in range(7):
            y0 = int(rng.integers(0, H - 60))
            hh = int(rng.integers(8, 60))
            a[y0:y0 + hh] = np.roll(a[y0:y0 + hh], int(rng.integers(-40, 40)), axis=1)
        a[..., 0] = np.roll(a[..., 0], 7, axis=1)
        a[..., 2] = np.roll(a[..., 2], -7, axis=1)
    return np.clip(a, 0, 255).astype(np.uint8).tobytes()


# =====================================================================
# 12. PROGRAMA PRINCIPAL
# =====================================================================
def main():
    global CTX
    t00 = time.time()
    os.makedirs(WORK, exist_ok=True)
    print("1/5 Dibujando las fotografías antiguas…")
    layers, M = [], []
    for i, Ly in enumerate(LAYOUT):
        lv, meta = hacer_foto(Ly["year"])
        layers.append(dict(i=i, levels=lv))
        M.append(meta)
    print("2/5 Narración con voz…")
    voice, ST, EN, t_cut, T = preparar_voz()
    G = construir_guion(ST, EN, M, t_cut)
    print("3/5 Música y efectos de sonido…")
    wav = hacer_audio(voice, ST, EN, T, G["t_rev"], t_cut, G["sfx"])
    print("4/5 Preparando escenario…")
    rng = np.random.default_rng(5)
    ln, ld = hacer_luz()
    CTX = dict(cam=G["cam"], ev=G["ev"], flick=G["flick"], dark=G["dark"], t_rev=G["t_rev"], t_cut=t_cut,
               t_s5=G["t_s5"], layers=layers, bg=hacer_mesa(), luz_n=ln, luz_d=ld,
               noise=[(rng.normal(0, 5.5, (H, W, 1))).astype(np.int16) for _ in range(6)],
               motes=[(rng.random() * W, rng.random() * H, .3 + .7 * rng.random(), 1.2 + 1.6 * rng.random()) for _ in range(26)],
               black=np.zeros((H, W, 3), np.uint8).tobytes())
    N = int(round(T * FPS))
    if "--prueba" in sys.argv:
        return prueba(T, ST, EN, G)
    print("5/5 Renderizando %d fotogramas (%.1f s de video). Tarda unos minutos…" % (N, T))
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
           "-movflags", "+faststart", "-shortest", SALIDA]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ncpu = max(1, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1))
    t1 = time.time()

    def prog(k):
        if k % 90 == 0 and k:
            el = time.time() - t1
            print("   %d%%  (faltan ~%d s)" % (100 * k // N, el / k * (N - k)), flush=True)

    if ncpu > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(ncpu) as pool:
            for k, fr in enumerate(pool.imap(render_frame, range(N), chunksize=3)):
                proc.stdin.write(fr)
                prog(k)
    else:
        for k in range(N):
            proc.stdin.write(render_frame(k))
            prog(k)
    proc.stdin.close()
    proc.wait()
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", SALIDA],
                         capture_output=True, text=True).stdout.strip()
    print("\n✅ LISTO: %s  (%.1f s) — tardó %.0f s en total" % (SALIDA, float(out or 0), time.time() - t00))
    return SALIDA


def prueba(T, ST, EN, G):
    """Modo de prueba: guarda fotogramas sueltos para revisarlos."""
    tiempos = [float(x) for x in os.environ.get("TIEMPOS", "0.3,3,6,10,14,19,24,30,35,40").split(",")]
    for tt in tiempos:
        i = int(tt * FPS)
        fr = render_frame(i)
        Image.frombytes("RGB", (W, H), fr).save(os.path.join(WORK, "f_%05.1f.png" % tt))
    print("tiempos:", {k: round(v, 1) for k, v in ST.items()}, "T=%.1f cut=%.1f" % (T, G["t_cut"]))


if __name__ == "__main__":
    resultado = main()
    try:
        from google.colab import files
        files.download(SALIDA)
    except Exception:
        pass
