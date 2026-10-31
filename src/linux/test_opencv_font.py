"""Test OpenCV text rendering on RK3576 Linux: putText (ASCII), freetype (Chinese),
and mixed rendering to work around fonts that lack ASCII glyphs.

Images saved to output/ — headless-safe.

render  = blank canvas, verify text characters render correctly in isolation
overlay = simulated dark camera frame, verify text looks right on a real feed
"""

import cv2
import numpy as np
import os
import glob
import time

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

CANVAS_W, CANVAS_H = 720, 160

# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

CJK_FONTS = [
    # WenQuanYi (most common on Debian/Ubuntu ARM)
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    # Noto CJK (full coverage: CJK + Latin + digits)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    # Droid
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/droid/DroidSansJapanese.ttf",
    # Arphic (uming = Ming/Song, ukai = Kai)
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    # Source Han / Noto alternative paths
    "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
    "/usr/share/fonts/truetype/source-han-sans/SourceHanSansSC-Regular.otf",
    # Firefly
    "/usr/share/fonts/truetype/firefly/fireflysung.ttf",
    # Hana (Japanese but covers many CJK)
    "/usr/share/fonts/truetype/hanazono/HanaMinA.ttf",
    # Generic CJK
    "/usr/share/fonts/truetype/cjk/unifont.ttf",
    # Debian/Ubuntu fallback
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


# ASCII-only fonts that look good paired with CJK fonts
ASCII_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def find_cjk_font():
    """Return first existing CJK font path."""
    font = _first_existing(CJK_FONTS)
    if font:
        return font
    # Broader glob fallback
    for pattern in [
        "/usr/share/fonts/**/*.ttc",
        "/usr/share/fonts/**/*.ttf",
        "/usr/share/fonts/**/*.otf",
    ]:
        for p in glob.glob(pattern, recursive=True):
            return p
    return None


def find_ascii_font():
    """Return first existing ASCII/Latin font path (for optional pairing)."""
    font = _first_existing(ASCII_FONTS)
    if font:
        return font
    # Fallback: any ttf that isn't the CJK one
    cjk = find_cjk_font()
    for pattern in [
        "/usr/share/fonts/**/*.ttf",
        "/usr/share/fonts/**/*.otf",
    ]:
        for p in glob.glob(pattern, recursive=True):
            if p != cjk:
                return p
    return None


def list_all_fonts():
    """List every font file found on the system."""
    fonts = []
    for pattern in [
        "/usr/share/fonts/**/*.ttc",
        "/usr/share/fonts/**/*.ttf",
        "/usr/share/fonts/**/*.otf",
    ]:
        fonts.extend(glob.glob(pattern, recursive=True))
    return sorted(fonts)


# ---------------------------------------------------------------------------
# Mixed render: ASCII → putText, CJK → freetype — zero frame-copy
# ---------------------------------------------------------------------------

def _needs_freetype(c):
    """True if cv2.putText Hershey fonts can't render this character.

    Hershey fonts only cover basic printable ASCII (0x20–0x7E).
    Everything else — CJK, symbols, degree sign, Greek, etc. — needs freetype.
    """
    cp = ord(c)
    return not (0x20 <= cp <= 0x7E)


def put_mixed_text(img, text, org, ft, font_height=24, color=(0, 255, 0),
                   thickness=2, line_type=cv2.LINE_AA):
    """Draw `text` on `img` in-place.

    Basic printable ASCII (0x20–0x7E) → cv2.putText (fast, no font file).
    Everything else (CJK, °, ±, ×, Greek, etc.) → freetype via CJK font.
    Zero frame-copy.

    Works with any CJK font — even those that lack Latin glyphs entirely.
    """
    x, y = org
    i = 0
    while i < len(text):
        if _needs_freetype(text[i]):
            j = i
            while j < len(text) and _needs_freetype(text[j]):
                j += 1
            segment = text[i:j]
            ft.putText(img, segment, (x, y),
                       fontHeight=font_height, color=color,
                       thickness=thickness, line_type=line_type,
                       bottomLeftOrigin=True)
            size, _ = ft.getTextSize(segment, fontHeight=font_height,
                                     thickness=thickness)
            x += size[0]
            i = j
        else:
            j = i
            while j < len(text) and not _needs_freetype(text[j]):
                j += 1
            segment = text[i:j]
            scale = font_height / 30.0
            cv2.putText(img, segment, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale,
                        color, thickness, line_type)
            (tw, _), _ = cv2.getTextSize(segment, cv2.FONT_HERSHEY_SIMPLEX,
                                         scale, thickness)
            x += tw
            i = j
    return img


# ---------------------------------------------------------------------------
# Tests: "render" = blank canvas / "overlay" = simulated camera frame
# ---------------------------------------------------------------------------

def test_put_text_ascii():
    """putText with pure ASCII — always works, no freetype needed."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.putText(canvas, "Ship Detection OK", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(canvas, "AIS Signal: Active", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    path = os.path.join(OUT_DIR, "puttext_ascii.jpg")
    cv2.imwrite(path, canvas)
    return path


def test_freetype_cn(font_path):
    """freetype-only on Chinese labels — tests if font has Latin glyphs."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName=font_path, id=0)

    ft.putText(canvas, "船舶检测 正常", (20, 40),
               fontHeight=36, color=(0, 255, 0), thickness=2,
               line_type=cv2.LINE_AA, bottomLeftOrigin=True)
    ft.putText(canvas, "AIS信号: 活跃", (20, 80),
               fontHeight=28, color=(255, 255, 0), thickness=2,
               line_type=cv2.LINE_AA, bottomLeftOrigin=True)

    path = os.path.join(OUT_DIR, "freetype_chinese.jpg")
    cv2.imwrite(path, canvas)
    return path


def test_mixed_render(font_path):
    """Mixed render on blank canvas — Chinese + ASCII blended."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName=font_path, id=0)

    put_mixed_text(canvas, "船舶检测 正常", (20, 45), ft,
                   font_height=36, color=(0, 255, 0))
    put_mixed_text(canvas, "AIS信号: 活跃", (20, 90), ft,
                   font_height=28, color=(255, 255, 0))
    path = os.path.join(OUT_DIR, "mixed_render.jpg")
    cv2.imwrite(path, canvas)
    return path


def test_comprehensive_overlay(font_path):
    """Overlay with Chinese + digits + punctuation + English, all mixed.

    This is the most realistic test: a simulated camera frame with a
    label containing every character type you'd see in production.
    """
    bg = np.zeros((520, 720, 3), dtype=np.uint8)
    bg[:] = (35, 35, 38)             # dark camera-like background
    cv2.rectangle(bg, (40, 60), (680, 380), (55, 55, 58), -1)  # "ROI"
    cv2.line(bg, (40, 60), (680, 60), (80, 80, 80), 1)

    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName=font_path, id=0)

    put_mixed_text(bg, "摄像头 #1 | 船舶: 3艘 | 航速: 12.5 kn", (30, 40),
                   ft, font_height=26, color=(0, 255, 255))
    put_mixed_text(bg, "AIS: MMSI=412345678  CPA=0.8nm  TCPA=12min", (30, 80),
                   ft, font_height=22, color=(200, 200, 200))
    put_mixed_text(bg, "位置: 30°25.7'N  122°46.3'E  航向: 045°", (30, 120),
                   ft, font_height=22, color=(200, 200, 200))
    put_mixed_text(bg, "告警: 靠近(0.5nm)  -- 触发阈值 < 1.0nm", (30, 480),
                   ft, font_height=22, color=(0, 200, 255))
    put_mixed_text(bg, "时间:2026-05-25 14:32:08 | 帧率:25fps | 状态:OK", (30, 160),
                   ft, font_height=20, color=(150, 150, 150))

    path = os.path.join(OUT_DIR, "comprehensive_overlay.jpg")
    cv2.imwrite(path, bg)
    return path


def test_mixed_overlay(font_path):
    """Simpler overlay with just Chinese + ASCII."""
    bg = np.zeros((480, 640, 3), dtype=np.uint8)
    bg[:] = (40, 40, 40)
    cv2.rectangle(bg, (80, 100), (560, 300), (60, 60, 60), -1)

    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName=font_path, id=0)

    put_mixed_text(bg, "摄像头 #1  船舶: 3  航速: 12 kn", (30, 55), ft,
                   font_height=24, color=(0, 255, 255))
    put_mixed_text(bg, "告警: 无", (30, 440), ft,
                   font_height=22, color=(0, 255, 0))

    path = os.path.join(OUT_DIR, "mixed_overlay.jpg")
    cv2.imwrite(path, bg)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("OpenCV Version:", cv2.__version__)
    print("freetype available:", hasattr(cv2, "freetype"))
    print("Output dir:", OUT_DIR)
    print("=" * 60)

    # 1. List all available fonts
    all_fonts = list_all_fonts()
    print(f"\nAll fonts found ({len(all_fonts)}):")
    for f in all_fonts:
        marker = ""
        if f == find_cjk_font():
            marker = " <-- CJK (selected)"
        print(f"  {f}{marker}")
    print()

    # 2. putText ASCII (no freetype needed)
    path = test_put_text_ascii()
    print(f"[putText ASCII]  PASS -> {path}")

    # 3. Freetype tests
    font_path = find_cjk_font()
    ascii_font = find_ascii_font()

    if font_path is None:
        print("[freetype] SKIP — no CJK font found.")
        print("  Install: apt install fonts-wqy-zenhei")
    elif not hasattr(cv2, "freetype"):
        print("[freetype] SKIP — cv2.freetype not available.")
    else:
        print(f"\nCJK font:  {font_path}")
        print(f"ASCII font: {ascii_font or '(none found)'}")
        print(f"Note: put_mixed_text only needs the CJK font — "
              "ASCII chars use cv2.putText directly.\n")

        tests = [
            ("freetype-only CN     ", test_freetype_cn),
            ("mixed render         ", test_mixed_render),
            ("mixed overlay        ", test_mixed_overlay),
            ("comprehensive overlay", test_comprehensive_overlay),
        ]

        for label, fn in tests:
            try:
                t0 = time.time()
                path = fn(font_path)
                elapsed = (time.time() - t0) * 1000
                print(f"  [{label}] PASS ({elapsed:.1f} ms) -> {path}")
            except Exception as e:
                print(f"  [{label}] FAIL: {e}")

    print("\nDone.")
