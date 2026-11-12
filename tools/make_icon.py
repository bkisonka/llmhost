# Генерирует src/app.ico: тёмный скруглённый квадрат с зелёным пузырём чата.
# Чистый Python: рисуем в 768x768, усредняем до 16/32/48, пишем PNG-in-ICO.
# Запуск: python tools/make_icon.py
import struct, zlib, os

S = 768
BG = (24, 27, 18)       # тёмный, как фон приложения
FG = (122, 200, 72)     # зелёный акцент
DOT = (24, 27, 18)

def in_rounded(x, y, x0, y0, x1, y1, r):
    if x < x0 or x > x1 or y < y0 or y > y1:
        return False
    dx = max(x0 + r - x, x - (x1 - r), 0)
    dy = max(y0 + r - y, y - (y1 - r), 0)
    return dx * dx + dy * dy <= r * r

def in_circle(x, y, cx, cy, r):
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r

def in_triangle(x, y, ax, ay, bx, by, cx, cy):
    def sign(px, py, qx, qy, rx, ry):
        return (px - rx) * (qy - ry) - (qx - rx) * (py - ry)
    d1 = sign(x, y, ax, ay, bx, by)
    d2 = sign(x, y, bx, by, cx, cy)
    d3 = sign(x, y, cx, cy, ax, ay)
    neg = d1 < 0 or d2 < 0 or d3 < 0
    pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (neg and pos)

def pixel(x, y):
    """Цвет пикселя canvas (0..S); альфа 0..255."""
    u, v = x / S, y / S
    # фон — скруглённый квадрат
    if not in_rounded(u, v, 0.02, 0.02, 0.98, 0.98, 0.22):
        return (0, 0, 0, 0)
    # пузырь + хвост
    if in_rounded(u, v, 0.14, 0.16, 0.86, 0.60, 0.12) or \
       in_triangle(u, v, 0.26, 0.58, 0.26, 0.80, 0.48, 0.58):
        # три точки-реплики
        for cx in (0.32, 0.50, 0.68):
            if in_circle(u, v, cx, 0.38, 0.052):
                return DOT + (255,)
        return FG + (255,)
    return BG + (255,)

def downsample(target):
    k = S // target
    rows = []
    for ty in range(target):
        row = bytearray()
        for tx in range(target):
            r = g = b = a = 0
            for sy in range(k):
                for sx in range(k):
                    pr, pg, pb, pa = pixel(tx * k + sx, ty * k + sy)
                    r += pr; g += pg; b += pb; a += pa
            n = k * k
            row += bytes((r // n, g // n, b // n, a // n))
        rows.append(bytes(row))
    return rows

def png(rows, w):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, w, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + r for r in rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))

imgs = []
for size in (16, 32, 48):
    data = png(downsample(size), size)
    imgs.append((size, data))

out = struct.pack("<HHH", 0, 1, len(imgs))
offset = 6 + 16 * len(imgs)
for size, data in imgs:
    out += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
    offset += len(data)
for _, data in imgs:
    out += data

path = os.path.join(os.path.dirname(__file__), "..", "src", "app.ico")
with open(path, "wb") as f:
    f.write(out)
print("OK:", os.path.abspath(path), len(out), "bytes")
