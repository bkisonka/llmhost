#!/usr/bin/env python3
# Сшить shared-MTP-голову с тензорами ствола: в файл головы добавляются
# отсутствующие тензоры (token_embd.weight), данные копируются из частей
# ствола, офсеты существующих тензоров пересчитываются под новый заголовок.
# Флаг nextn_shared_target_tensors снимается — голова грузится как обычная.
#
# Запуск: merge_shared_head.py <shared.gguf> <out.gguf> <тензор[,тензор…]> <glob частей ствола>
import struct, sys, os, glob

SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def rd_str(f):
    n, = struct.unpack('<Q', f.read(8)); return f.read(n).decode('utf-8','replace')

def parse(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver, = struct.unpack('<I', f.read(4))
    ntens, = struct.unpack('<Q', f.read(8))
    nkv, = struct.unpack('<Q', f.read(8))
    kvs = []
    align = 32
    for _ in range(nkv):
        k = rd_str(f)
        t, = struct.unpack('<I', f.read(4))
        if t == 8:
            raw = f.read(8); n, = struct.unpack('<Q', raw); v = raw + f.read(n)
        elif t == 9:
            head = f.read(12)
            et, = struct.unpack('<I', head[:4]); n, = struct.unpack('<Q', head[4:])
            if SZ.get(et):
                v = head + f.read(n*SZ[et])
            elif et == 8:
                v = head
                for _ in range(n):
                    ln = f.read(8)
                    n2, = struct.unpack('<Q', ln)
                    v += ln + f.read(n2)
            else:
                raise SystemExit('вложенный массив')
        else:
            v = f.read(SZ[t])
        if k == 'general.alignment' and t == 4:
            align = struct.unpack('<I', v)[0]
        kvs.append([k, t, v])
    data0 = ((f.tell() + align - 1)//align)*align
    tens = []
    for _ in range(ntens):
        name = rd_str(f)
        nd, = struct.unpack('<I', f.read(4))
        dims = f.read(8*nd)
        tt, = struct.unpack('<I', f.read(4))
        off, = struct.unpack('<Q', f.read(8))
        tens.append([name, nd, dims, tt, off])
    f.close()
    return {'ver': ver, 'kvs': kvs, 'align': align, 'tens': tens,
            'data0': data0, 'fsize': os.path.getsize(path)}

# Байт/элемент по id типа — эмпирически с грузящихся файлов этого писателя
# (unsloth использует собственную нумерацию: 12 = Q4_K-плотность, 13 = Q5_K,
# 14 = Q6_K, 20 = IQ4_XS-плотность, 30 = BF16/F16).
RATIO = {0:(4,1),1:(1,1),2:(9,16),3:(5,8),6:(11,16),7:(3,4),8:(17,16),
         12:(9,16),13:(11,16),14:(105,128),20:(9,16),27:(9,16),30:(2,1),36:(2,1)}

def nbytes(tt, dims):
    num, den = RATIO[tt]
    n = 1
    for d in struct.unpack('<%dQ' % (len(dims)//8), dims):
        n *= d
    return n * num // den

def tensor_entry(name, nd, dims, tt, off):
    o = bytearray()
    o += struct.pack('<Q', len(name)) + name.encode()
    o += struct.pack('<I', nd) + dims + struct.pack('<I', tt) + struct.pack('<Q', off)
    return bytes(o)

head_path, out_path, missing_arg, trunk_glob = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
missing = missing_arg.split(',')
H = parse(head_path)
names = {t[0] for t in H['tens']}
add = [m for m in missing if m not in names]
print('тензоров в голове:', len(H['tens']), '| добавить из ствола:', add)

# Данные недостающих тензоров из частей ствола (размер спана — по соседям)
found = {}
for p in sorted(glob.glob(trunk_glob)):
    T = parse(p)
    if not T['tens']:
        continue
    ts = sorted(T['tens'], key=lambda t: t[4])
    f = open(p, 'rb')
    for i, t in enumerate(ts):
        if t[0] in add and t[0] not in found:
            end = ts[i+1][4] if i+1 < len(ts) else T['fsize'] - T['data0']
            f.seek(T['data0'] + t[4])
            found[t[0]] = (t[1], t[2], t[3], f.read(end - t[4]))
    f.close()
for m in add:
    assert m in found, 'в стволе не найден %s' % m

# KV: снять shared-флаг — голова больше не «shared»
H['kvs'] = [k for k in H['kvs'] if not k[0].endswith('.nextn_shared_target_tensors')]

# Заголовок и записи: движок требует плотную упаковку — первая запись с
# офсетом 0, дальше встык, данные в том же порядке, что записи. Пересобираем
# данные целиком: спаны старых тензоров (по отсортированным офсетам), порядок
# записей — исходный список головы + добавляемые в конце.
old_sorted = sorted(H['tens'], key=lambda t: t[4])
spans = {}
for i, t in enumerate(old_sorted):
    end = old_sorted[i+1][4] if i+1 < len(old_sorted) else H['fsize'] - H['data0']
    spans[t[0]] = end - t[4]
order = [t[0] for t in H['tens']] + add
hf = open(head_path, 'rb')
raw = {}
for name in order:
    if name in spans:
        src = next(t for t in old_sorted if t[0] == name)
        hf.seek(H['data0'] + src[4])
        raw[name] = hf.read(nbytes(src[3], src[2]))
    else:
        raw[name] = found[name][3]
hf.close()

entries = bytearray()
cur = 0
for name in order:
    if name in spans:
        t = next(t for t in H['tens'] if t[0] == name)
        entries += tensor_entry(name, t[1], t[2], t[3], cur)
    else:
        nd, dims, tt, _ = found[name]
        entries += tensor_entry(name, nd, dims, tt, cur)
    cur += len(raw[name])
data_len = cur

hdr = bytearray()
hdr += b'GGUF' + struct.pack('<I', H['ver'])
hdr += struct.pack('<Q', len(order))
hdr += struct.pack('<Q', len(H['kvs']))
for k, t, v in H['kvs']:
    hdr += struct.pack('<Q', len(k)) + k.encode() + struct.pack('<I', t) + v
ds = ((len(hdr) + len(entries) + H['align'] - 1)//H['align'])*H['align']

out = bytearray(hdr)
out += entries
out += b'\0' * (ds - len(out))
for name in order:
    out += raw[name]
open(out_path, 'wb').write(out)
print('собрано: %s (%.2f ГиБ), data0=%d, данные=%d МиБ' %
      (out_path, len(out)/2**30, ds, data_len >> 20))
