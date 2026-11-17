"""
ANALISIS PICK 4 — SATU FILE, SATU OUTPUT HTML

  Bagian 1 : HOT/OVERDUE + transisi Fireball→Ball + backtest (sesi D & E)
  Bagian 2 : Pola pasangan & shio — empat pasangan: Ball1+Ball2, Ball2+Ball3, Ball3+Ball4, Ball4+Fireball
  Bagian 3 : Statistik angka (semua posisi, Ball 1-4, Fireball) + shio semua pasangan (z-score)
             [BARU] Ringkasan gabungan tiap posisi (konsensus D · E · Gabungan) + ringkasan per posisi
             dengan highlight HOT / OVERDUE / irisan keduanya
  Bagian 4 : Histori Ball 1-4 — pertama muncul sampai terbaru, jeda terlama, semua data
             (tab "Semua data": terbaru di atas, angka tampil sebagai bola)

  Bagian 5 : Ball 1-4 — berapa kali angka muncul beruntun dalam 10 draw terbaru,
             plus kemunculan Ganjil/Genap dan Besar/Kecil (Besar = 5-9, Kecil = 0-4)
             (urutan D dan E digabung sesuai waktu; jumlah draw diatur lewat JUMLAH_DRAW_BERUNTUN)

Output HANYA satu file HTML (tanpa .db, .json, .txt, dan tanpa cetakan di konsol).

Kolom "Kode terkait" (tabel E) punya tombol "📋 Salin": klik tombol atau klik kodenya
untuk menyalin SEMUA kode sekaligus, formatnya:  07*19*31*43*55*67*79*91*

Contoh:
  python analisis_pick4_lengkap.py
  python analisis_pick4_lengkap.py data.csv
  python analisis_pick4_lengkap.py --bagian 2 --tambahan 12,47
  python analisis_pick4_lengkap.py --html laporan.html --mulai 2023-01-01
  python analisis_pick4_lengkap.py --hari senin               (hanya data hari Senin, semua bagian)
  python analisis_pick4_lengkap.py --bagian 4 --mulai ""      (histori dari data paling awal di CSV)
"""
import argparse
import html
import math
import os
import subprocess
import sys
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

# ==================================================
# KONFIGURASI
# ==================================================
CSV_LOKAL = r"E:\DATA-UTAMA\NCELPick4.csv"
CSV_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/NCELPick4.csv"
FOLDER_DEFAULT = r"E:\DATA-UTAMA"
NAMA_HTML = "hasil_analisis.html"
TANGGAL_MULAI = "2026-01-01"   # kosongkan ("") untuk memakai semua data di CSV

HARI = "semua"   # filter hari untuk SEMUA bagian: semua / senin / selasa / rabu / kamis / jumat / sabtu / minggu
HARI_NAMA = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]   # urutan = date.weekday()

TOP_DIGIT = 9
TOP_PRED = 9
MULAI_UJI = 107
TOP_PASANGAN = 25
TOP_SHIO = 12
TOP_KODE = 12
JUMLAH_ANGKA_HOT = 12
JUMLAH_PASANGAN_HOT = 12
JUMLAH_POLA_URUTAN = 12
Z_HOT = 1.0
P_OVERDUE = 0.05
BATAS_DOMINAN = 0.7          # sisi ganjil/genap atau besar/kecil disorot bila porsinya >= 70% dari draw yang diperiksa
JUMLAH_DRAW_BERUNTUN = 10   # Bagian 5: jumlah draw terbaru yang diperiksa untuk angka beruntun
TOP_OVERDUE_RINGKAS = 5   # overdue per posisi di ringkasan Bagian 3 (urut terlama absen)

# True  -> hasil salin: 07*19*31*43*55*67*79*91*   (ada * di ujung, sesuai contoh)
# False -> hasil salin: 07*19*31*43*55*67*79*91
AKHIRI_BINTANG = True

SHIO_LIST = [
    {"no": 1,  "kode": ["01","13","25","37","49","61","73","85","97"]},
    {"no": 2,  "kode": ["02","14","26","38","50","62","74","86","98"]},
    {"no": 3,  "kode": ["03","15","27","39","51","63","75","87","99"]},
    {"no": 4,  "kode": ["04","16","28","40","52","64","76","88","00"]},
    {"no": 5,  "kode": ["05","17","29","41","53","65","77","89"]},
    {"no": 6,  "kode": ["06","18","30","42","54","66","78","90"]},
    {"no": 7,  "kode": ["07","19","31","43","55","67","79","91"]},
    {"no": 8,  "kode": ["08","20","32","44","56","68","80","92"]},
    {"no": 9,  "kode": ["09","21","33","45","57","69","81","93"]},
    {"no": 10, "kode": ["10","22","34","46","58","70","82","94"]},
    {"no": 11, "kode": ["11","23","35","47","59","71","83","95"]},
    {"no": 12, "kode": ["12","24","36","48","60","72","84","96"]},
]
KODE2_SHIO = {k: s["no"] for s in SHIO_LIST for k in s["kode"]}
SHIO_NAMA = {s["no"]: f"Shio {s['no']:02d}" for s in SHIO_LIST}
SHIO_PROB = {s["no"]: len(s["kode"]) / 100 for s in SHIO_LIST}
assert len(KODE2_SHIO) == 100, "Peta shio harus mencakup 00-99 tepat sekali"

KOLOM_ANGKA = ["Ball 1", "Ball 2", "Ball 3", "Ball 4"]
KOLOM_FB = {'b1': 'Ball 1', 'b2': 'Ball 2', 'b3': 'Ball 3', 'b4': 'Ball 4', 'fb': 'Fireball'}
NAMA_FB = {'b1': 'BALL 1', 'b2': 'BALL 2', 'b3': 'BALL 3', 'b4': 'BALL 4', 'fb': 'FIREBALL'}
ETIKET = [('b1', 'FB→B1'), ('b2', 'FB→B2'), ('b3', 'FB→B3'), ('b4', 'FB→B4'), ('fb', 'FB→FB')]

# Pasangan: indeks Ball (0-based) atau "fb" untuk Kepala (a) & Ekor (b)
PAIRS = {
    "12": {"a": 0, "b": 1, "kepala": "Ball 1", "ekor": "Ball 2", "label": "Ball 1 + Ball 2"},
    "23": {"a": 1, "b": 2, "kepala": "Ball 2", "ekor": "Ball 3", "label": "Ball 2 + Ball 3"},
    "34": {"a": 2, "b": 3, "kepala": "Ball 3", "ekor": "Ball 4", "label": "Ball 3 + Ball 4"},
    "4F": {"a": 3, "b": "fb", "kepala": "Ball 4", "ekor": "Fireball", "label": "Ball 4 + Fireball"},
}


def nama_shio(no):
    return SHIO_NAMA.get(no, "-") if no is not None else "-"


def nilai_pos(r, p):
    """Nilai posisi p pada rekaman r: indeks Ball (0-3) atau 'fb' untuk Fireball."""
    return r["fb"] if p == "fb" else r["b"][p]


# ==================================================
# FUNGSI UMUM
# ==================================================
def hot_overdue(nilai, semesta, top):
    """Baris = (nilai, jumlah, jarak). Mengembalikan (hot, overdue, persamaan)."""
    total = len(nilai)
    cnt = Counter(nilai)
    terakhir = {}
    for i, v in enumerate(nilai):
        terakhir[v] = i
    baris = [(v, cnt.get(v, 0), (total - 1 - terakhir[v]) if v in terakhir else total) for v in semesta]
    hot = sorted(baris, key=lambda x: (-x[1], x[0]))[:top]
    ovd = sorted(baris, key=lambda x: (-x[2], x[0]))[:top]
    ovd_set = {o[0] for o in ovd}
    sama = [h[0] for h in hot if h[0] in ovd_set]
    return hot, ovd, sama


def jarak_min(pos):
    return min(b - a for a, b in zip(pos, pos[1:])) if len(pos) > 1 else None


# ==================================================
# PEMBANTU HTML
# ==================================================
class Raw(str):
    """Teks yang sudah berupa HTML (tidak di-escape lagi)."""


def e(x):
    return html.escape(str(x))


def sel(c):
    return c if isinstance(c, Raw) else e(c)


def tbl(head, rows, baris_kls=None, tinggi=False):
    """baris_kls: daftar kelas CSS per baris (opsional). tinggi=True -> tabel dibatasi tinggi & bisa digulir."""
    h = "".join(f"<th>{e(x)}</th>" for x in head)
    b = ""
    for i, r in enumerate(rows):
        k = baris_kls[i] if baris_kls and i < len(baris_kls) and baris_kls[i] else ""
        b += f'<tr class="{k}">' + "".join(f"<td>{sel(c)}</td>" for c in r) + "</tr>"
    return (f'<div class="scroll{" tall" if tinggi else ""}"><table>'
            f'<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')


def chips(lst, kls=""):
    return Raw("".join(f'<span class="chip {kls}">{e(x)}</span>' for x in lst) or '<span class="kosong">—</span>')


def chip_info(nilai, info="", kls="", kuat=False, judul=""):
    """Chip dengan angka tebal + keterangan kecil (mis. frekuensi / absen)."""
    ttl = f' title="{e(judul)}"' if judul else ""
    kecil = f"<small>{e(info)}</small>" if info != "" else ""
    return (f'<span class="chip {kls}{" kuat" if kuat else ""}"{ttl}><b>{e(nilai)}</b>{kecil}</span>')


def badge(status):
    return Raw(f'<span class="badge {status.lower()}">{e(status)}</span>')


def kode_salin(daftar_kode):
    """Sel tabel berisi kode + tombol salin.
    Teks tampil: 07*19*31 ; teks yang disalin: 07*19*31*  (tergantung AKHIRI_BINTANG)."""
    daftar_kode = list(daftar_kode)
    if not daftar_kode:
        return Raw("—")
    tampil = "*".join(daftar_kode)
    salin = tampil + ("*" if AKHIRI_BINTANG else "")
    return Raw(f'<span class="kode" data-copy="{e(salin)}" title="Klik untuk menyalin semua">{e(tampil)}</span> '
               f'<button type="button" class="copy" data-copy="{e(salin)}">📋 Salin</button>')


def card(judul, isi, kls=""):
    return f'<section class="card {kls}"><h3>{e(judul)}</h3>{isi}</section>'


def grid(*isi):
    return '<div class="grid">' + "".join(isi) + "</div>"


def catatan_html(teks):
    return f'<p class="note">{e(teks)}</p>'


def legenda():
    return Raw('<p class="legenda"><span class="chip hot">HOT</span> sering muncul '
               '<span class="chip ovd">OVERDUE</span> lama absen '
               '<span class="chip sama">HOT∩OVERDUE</span> irisan keduanya '
               '<span class="chip pred">POLA</span> pola urutan</p>')


PREFIX = ""   # awalan id tab; diisi saat membangun tiap pilihan hari supaya id tidak bentrok


def tabs(gid, items, kls=""):
    gid = PREFIX + gid
    nav = "".join(f'<button class="{"on" if i == 0 else ""}" data-g="{gid}" data-t="{gid}-{i}">{e(l)}</button>'
                  for i, (l, _) in enumerate(items))
    panels = "".join(f'<div class="panel {"on" if i == 0 else ""}" id="{gid}-{i}" data-g="{gid}">{h}</div>'
                     for i, (_, h) in enumerate(items))
    return f'<nav class="tabs {kls}">{nav}</nav>{panels}'


# ==================================================
# PEMUAT DATA
# ==================================================
def muat_data(sumber, mulai=TANGGAL_MULAI):
    catatan = []
    df = pd.read_csv(sumber, quotechar='"', skip_blank_lines=True, on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]
    for wajib in ["Date", "Day/Eve"] + KOLOM_ANGKA:
        if wajib not in df.columns:
            sys.exit(f"Kolom '{wajib}' tidak ada di CSV. Kolom tersedia: {list(df.columns)}")
    if "Fireball" not in df.columns:
        df["Fireball"] = np.nan

    tgl = df["Date"].astype(str).str.strip()
    d1 = pd.to_datetime(tgl, format="%m/%d/%Y", errors="coerce")
    d2 = pd.to_datetime(tgl, format="%Y-%m-%d", errors="coerce")
    df["Date"] = d1.fillna(d2)

    if mulai:
        try:
            batas = pd.Timestamp(mulai)
        except Exception:
            sys.exit(f"Format tanggal mulai salah: '{mulai}' (pakai YYYY-MM-DD)")
        df = df[df["Date"] >= batas]
    awal = len(df)

    df["Day/Eve"] = df["Day/Eve"].astype(str).str.strip().str.upper()
    df = df[df["Day/Eve"].isin(["D", "E"])]
    for k in KOLOM_ANGKA + ["Fireball"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["Date"] + KOLOM_ANGKA)
    df = df[((df[KOLOM_ANGKA] >= 0) & (df[KOLOM_ANGKA] <= 9) & (df[KOLOM_ANGKA] % 1 == 0)).all(axis=1)]
    for k in KOLOM_ANGKA:
        df[k] = df[k].astype(int)
    fb = df["Fireball"]
    df["Fireball"] = fb.where(fb.between(0, 9) & (fb % 1 == 0))

    sebelum = len(df)
    df = df.drop_duplicates(subset=["Date", "Day/Eve"], keep="last")
    duplikat = sebelum - len(df)
    df = df.sort_values(["Date", "Day/Eve"]).reset_index(drop=True)

    dibuang = awal - len(df) - duplikat
    if dibuang:
        catatan.append(f"{dibuang} baris dibuang (sesi rusak, Ball kosong, atau di luar 0-9).")
    if duplikat:
        catatan.append(f"{duplikat} baris duplikat dibuang.")
    if df.empty:
        sys.exit(f"Tidak ada data valid mulai {mulai} — cek format CSV (MM/DD/YYYY atau YYYY-MM-DD)")
    return df[["Date", "Day/Eve"] + KOLOM_ANGKA + ["Fireball"]], catatan


def buat_recs(df):
    return [{
        "tanggal": r["Date"].strftime("%Y-%m-%d"),
        "tipe": r["Day/Eve"],
        "hari": r["Date"].weekday(),
        "slot": r["Date"].toordinal() * 2 + (0 if r["Day/Eve"] == "D" else 1),
        "b": tuple(int(r[k]) for k in KOLOM_ANGKA),
        "fb": int(r["Fireball"]) if pd.notna(r["Fireball"]) else None,
    } for r in df.to_dict("records")]


def buat_view(recs, pk):
    """Tambahkan kode/shio/kepala/ekor untuk pasangan pk. Rekaman dengan nilai None
    (Fireball tidak valid) dilewati; 'idx' menyimpan indeks asli di recs."""
    P = PAIRS[pk]
    out = []
    for i, r in enumerate(recs):
        a, b = nilai_pos(r, P["a"]), nilai_pos(r, P["b"])
        if a is None or b is None:
            continue
        kode = f"{a}{b}"
        s = KODE2_SHIO[kode]
        out.append({**r, "idx": i, "kode": kode, "shio_no": s, "shio": SHIO_NAMA[s],
                    "kepala": a, "ekor": b, "kepekor": f"{a}-{b}",
                    "digit4": "".join(map(str, r["b"]))})
    return out


# ==================================================
# BAGIAN 1 — HOT/OVERDUE + FIREBALL→BALL + BACKTEST
# ==================================================
def bangun_matrik(asal, tujuan, batas):
    m = np.zeros((10, 10), dtype=int)
    for k in range(batas - 1):
        m[asal[k], tujuan[k + 1]] += 1
    s = m.sum(1, keepdims=True)
    p = np.divide(m, s, out=np.zeros(m.shape, dtype=float), where=s != 0)
    return m, p


def backtest(asal, tujuan):
    n = len(asal)
    if n <= MULAI_UJI:          # data terlalu sedikit untuk backtest (mis. filter hari)
        return 0, 0, 0
    m = np.zeros((10, 10), dtype=int)
    for k in range(MULAI_UJI - 1):
        m[asal[k], tujuan[k + 1]] += 1
    uji = t1 = t3 = 0
    for i in range(MULAI_UJI, n - 1):
        m[asal[i - 1], tujuan[i]] += 1
        row = m[asal[i]]
        if row.sum() == 0:
            continue
        urut = sorted(range(10), key=lambda j: (-row[j], j))
        benar = tujuan[i + 1]
        uji += 1
        t1 += urut[0] == benar
        t3 += benar in urut[:3]
    return uji, t1, t3


def pct(a, b):
    return f"{a / b * 100:.1f}%" if b else "Data kurang"


def analisis_fb(dfs):
    n = len(dfs)
    seri = {k: dfs[c].astype(int).tolist() for k, c in KOLOM_FB.items()}
    fb = seri['fb']
    last = dfs.iloc[-1]
    hasil = {
        'n': n,
        'terakhir': {'tanggal': last['Date'].strftime('%d/%m/%Y'),
                     'bola': ''.join(str(int(last[KOLOM_FB[k]])) for k in ('b1', 'b2', 'b3', 'b4')),
                     'fb': int(last['Fireball'])},
        'hot': {}, 'ovd': {}, 'sama': {}, 'pred': {}, 'uji': {},
    }
    for k in KOLOM_FB:
        hot, ovd, sama = hot_overdue(seri[k], range(10), TOP_DIGIT)
        hasil['hot'][k] = [h[0] for h in hot]
        hasil['ovd'][k] = [o[0] for o in ovd]
        hasil['sama'][k] = sama
        _, p = bangun_matrik(fb, seri[k], n)
        row = p[fb[-1]]
        urut = sorted(range(10), key=lambda j: (-row[j], j))[:TOP_PRED]
        hasil['pred'][k] = urut if row.sum() > 0 else []
        u, t1, t3 = backtest(fb, seri[k])
        hasil['uji'][k] = (u, pct(t1, u), pct(t3, u))
    return hasil


def kartu_fb(H, judul):
    t = H['terakhir']
    isi = (f'<p class="meta">Terakhir: {e(t["tanggal"])} · '
           f'<span class="bola kecil">{e(t["bola"][0])}</span>'
           f'<span class="bola kecil">{e(t["bola"][1])}</span>'
           f'<span class="bola kecil">{e(t["bola"][2])}</span>'
           f'<span class="bola kecil">{e(t["bola"][3])}</span> '
           f'<span class="bola kecil fb">{t["fb"]}</span> · Sampel: {H["n"]}</p>')
    isi += tbl(["Posisi", "🔥 HOT", "⏳ OVERDUE", "✅ Persamaan"],
               [(NAMA_FB[k], chips(H['hot'][k], 'hot'), chips(H['ovd'][k], 'ovd'), chips(H['sama'][k], 'sama'))
                for k in KOLOM_FB])
    isi += tbl(["Transisi", "🎯 Prediksi", "Uji", "Top1", "Top3"],
               [(lbl, chips(H['pred'][k], 'pred'), f"{H['uji'][k][0]} kali", H['uji'][k][1], H['uji'][k][2])
                for k, lbl in ETIKET])
    pers = []
    for k, lbl in ETIKET:
        pred = H['pred'][k]
        p_hot = [d for d in pred if d in H['hot'][k]]
        p_ovd = [d for d in pred if d in H['ovd'][k]]
        p_dua = [d for d in p_hot if d in H['ovd'][k]]
        pers.append((lbl, chips(p_hot, 'hot'), chips(p_ovd, 'ovd'), chips(p_dua, 'sama')))
    isi += tbl(["Transisi", "✅ Prediksi ∩ HOT", "✅ Prediksi ∩ OVERDUE", "✅ Prediksi ∩ HOT ∩ OVERDUE"], pers)
    return card(judul, isi)


def bagian1(df):
    dfb = df[df["Fireball"].notna()].copy()
    pre = ""
    if len(dfb) < len(df):
        pre = catatan_html(f"{len(df) - len(dfb)} baris tanpa Fireball valid tidak dipakai di Bagian 1 "
                           f"(urutan transisi di sekitarnya bisa tidak utuh).")
    dfb["Fireball"] = dfb["Fireball"].astype(int)
    df_d = dfb[dfb["Day/Eve"] == "D"].reset_index(drop=True)
    df_e = dfb[dfb["Day/Eve"] == "E"].reset_index(drop=True)
    if len(df_d) < 2 or len(df_e) < 2:
        return pre + catatan_html("Data Fireball sesi D/E tidak cukup — Bagian 1 dilewati.")
    D, E = analisis_fb(df_d), analisis_fb(df_e)
    info = catatan_html("Pembanding kebetulan murni: Top1 ≈ 10%, Top3 ≈ 30%. "
                        "Hasil mendekati angka ini = tidak ada sinyal.")
    return pre + legenda() + grid(kartu_fb(D, "☀️ Sesi D"), kartu_fb(E, "🌙 Sesi E")) + info


# ==================================================
# BAGIAN 2 — POLA PASANGAN & SHIO (D↔E), per pasangan
# ==================================================
def ho_card(judul, hot, ovd, sama):
    rows = []
    for i in range(max(len(hot), len(ovd))):
        rows.append((hot[i][0] if i < len(hot) else "", hot[i][1] if i < len(hot) else "",
                     ovd[i][0] if i < len(ovd) else "", f"{ovd[i][2]} undian" if i < len(ovd) else ""))
    isi = tbl(["🔥 Nilai", "Jumlah", "⏳ Nilai", "Tidak muncul"], rows)
    if sama:
        isi += ('<p class="meta">✅ Persamaan HOT∩OVERDUE: '
                + "".join(f'<span class="chip sama">{e(s)}</span>' for s in sama) + "</p>")
    else:
        isi += '<p class="meta">✅ Persamaan HOT∩OVERDUE: —</p>'
    return card(judul, isi)


def bagian2(recs, pk, tambahan_input):
    P = PAIRS[pk]
    N = len(recs)
    if N < 2:
        return catatan_html("Data valid kurang dari 2 undian untuk pasangan ini — bagian ini dilewati.")
    out = []
    if pk == "4F":
        out.append(catatan_html("Pasangan ini memakai Fireball; undian tanpa Fireball valid dilewati."))

    tgl_terakhir = recs[-1]["tanggal"]
    data_D = next((r for r in recs if r["tanggal"] == tgl_terakhir and r["tipe"] == "D"), None)
    data_E = next((r for r in recs if r["tanggal"] == tgl_terakhir and r["tipe"] == "E"), None)

    def blok(r, label):
        if not r:
            return card(label, catatan_html("Tidak ada data"))
        return card(f"{label} | {r['tanggal']}", tbl(["Item", "Nilai"], [
            ("4 Digit", r["digit4"]),
            ("Fireball", r["fb"] if r["fb"] is not None else "-"),
            ("Pasangan", f"{r['kode']} ({P['label']})"),
            (f"KEPALA ← {P['kepala']}", r["kepala"]),
            (f"EKOR ← {P['ekor']}", r["ekor"]),
            ("KEP-EKOR", r["kepekor"]),
            ("Shio", r["shio"]),
            ("Ball 1-4", " ".join(map(str, r["b"]))),
        ]))
    out.append("<h4>📋 Data terakhir</h4>" + grid(blok(data_D, "🌞 Siang (D)"), blok(data_E, "🌙 Sore (E)")))

    terakhir = recs[-1]
    tipe_terakhir = terakhir["tipe"]
    arah = "D" if tipe_terakhir == "E" else "E"
    label_arah = f"{tipe_terakhir} → {arah}"
    tambahan = [str(p).zfill(2) for p in tambahan_input if str(p).zfill(2) in KODE2_SHIO]
    daftar_cari = list(dict.fromkeys([terakhir["kode"]] + tambahan))
    daftar_shio_cari = list(dict.fromkeys([KODE2_SHIO[p] for p in daftar_cari]))

    valid_idx = [i for i in range(N - 1)
                 if recs[i]["tipe"] == tipe_terakhir and recs[i + 1]["tipe"] == arah
                 and recs[i + 1]["slot"] - recs[i]["slot"] == 1]

    out.append("<h4>📊 Ringkasan</h4>" + tbl(["Item", "Nilai"], [
        ("Pasangan", f"{P['label']} (Kepala = {P['kepala']}, Ekor = {P['ekor']})"),
        ("Total undian", N),
        ("Tanggal", terakhir["tanggal"]),
        ("Acuan terakhir", label_arah),
        ("Kepala / Ekor terakhir", f"{terakhir['kepala']} / {terakhir['ekor']}"),
        ("Pasangan terakhir", f"{terakhir['kode']} | {terakhir['shio']}"),
        (f"Transisi {label_arah} valid", len(valid_idx)),
        ("Pantau tambahan", ", ".join(tambahan) if tambahan else "—"),
    ]))

    out.append("<h4>A–C. HOT 🔥 &amp; OVERDUE ⏳</h4>" + grid(
        ho_card(f"A. KEPALA ({P['kepala']})", *hot_overdue([r["kepala"] for r in recs], range(10), TOP_DIGIT)),
        ho_card(f"B. EKOR ({P['ekor']})", *hot_overdue([r["ekor"] for r in recs], range(10), TOP_DIGIT)),
        ho_card(f"C. PASANGAN ({P['label']})",
                *hot_overdue([r["kode"] for r in recs], [f"{i:02d}" for i in range(100)], TOP_PASANGAN))))

    # D. Pola pasangan
    bagian_d = []
    for asal in daftar_cari:
        tanda = "⭐ Acuan" if asal == terakhir["kode"] else ""
        pos_per_tujuan = defaultdict(list)
        for i in valid_idx:
            if recs[i]["kode"] == asal:
                pos_per_tujuan[recs[i + 1]["kode"]].append(i)
        judul = f"Dari {asal} {tanda}".strip()
        if not pos_per_tujuan:
            bagian_d.append(card(judul, catatan_html(f"Belum ada riwayat {label_arah} untuk pasangan ini")))
            continue
        rows = [(ke, len(p), jarak_min(p)) for ke, p in pos_per_tujuan.items()]
        ada = sorted([r for r in rows if r[1] >= 2], key=lambda x: (-x[1], x[2]))
        sekali = sorted([r for r in rows if r[1] == 1], key=lambda x: x[0])
        jp = min((r[2] for r in ada), default=None)
        isi = ""
        if ada:
            isi += tbl(["Ke", "Shio tujuan", "Jlh", "Jarak min", ""],
                       [(ke, nama_shio(KODE2_SHIO[ke]), jml, f"{jm} undian",
                         "🏆 Jarak terpendek" if jm == jp else "") for ke, jml, jm in ada],
                       baris_kls=["r-hot" if jm == jp else "" for _, _, jm in ada])
        if sekali:
            isi += (f'<p class="meta">⚠️ Hanya 1 kali ({len(sekali)} pola): '
                    f'{e("*".join(ke for ke, _, _ in sekali))}</p>')
        bagian_d.append(card(judul, isi, "acuan" if asal == terakhir["kode"] else ""))
    out.append(f"<h4>D. Pola berdasarkan pasangan → {e(label_arah)}</h4>" + grid(*bagian_d))

    # E & F
    for shio_asal in daftar_shio_cari:
        idx = [i for i in valid_idx if recs[i]["shio_no"] == shio_asal]
        judul = f"E. Pola {nama_shio(shio_asal)} → {label_arah} | {TOP_SHIO} teratas | sampel: {len(idx)}"
        if not idx:
            out.append(card(judul, catatan_html("Tidak ada data")))
            continue
        pos_shio = defaultdict(list)
        kode_per_shio = defaultdict(Counter)
        kode_total = Counter()
        for i in idx:
            t = recs[i + 1]
            pos_shio[t["shio_no"]].append(i)
            kode_per_shio[t["shio_no"]][t["kode"]] += 1
            kode_total[t["kode"]] += 1
        urut = sorted(pos_shio.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:TOP_SHIO]
        peringkat = {}
        rows_e = []
        for no, (s, pos) in enumerate(urut, 1):
            peringkat[s] = no
            # daftar kode (urut sering→jarang) + tombol salin semua
            daftar_kode = [k for k, c in sorted(kode_per_shio[s].items(), key=lambda kv: (-kv[1], kv[0]))]
            jm = jarak_min(pos)
            rows_e.append((no, nama_shio(s), kode_salin(daftar_kode), f"{jm} undian" if jm else "Hanya 1 kali"))
        rows_f = [(kode, c, nama_shio(KODE2_SHIO[kode]), peringkat.get(KODE2_SHIO[kode], "-"))
                  for kode, c in sorted(kode_total.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_KODE]]
        out.append(grid(
            card(judul, tbl(["No", "Shio ke", "Kode terkait", "Jarak min"], rows_e)),
            card(f"F. Kode tersering setelah {nama_shio(shio_asal)} ({label_arah}) | top {TOP_KODE}",
                 tbl(["Kode", "Jlh", "Shio", "Urutan shio di E"], rows_f))))

    out.append(catatan_html("🔥 = paling sering | ⏳ = paling lama tidak muncul | "
                            "🏆 = jarak antar kemunculan terpendek"))
    return "".join(out)


# ==================================================
# BAGIAN 3 — STATISTIK ANGKA (Ball 1-4 + Fireball) + SHIO SEMUA PASANGAN
# ==================================================
def ambang_absen(p_hadir):
    if p_hadir <= 0 or p_hadir >= 1:
        return 10 ** 9
    return math.ceil(math.log(P_OVERDUE) / math.log(1 - p_hadir))


def jarak_absen(terakhir, total):
    return {k: (total - 1 - v if v is not None else total) for k, v in terakhir.items()}


def statistik_digit(data, posisi):
    total = len(data)
    frek = {n: 0 for n in range(10)}
    terakhir = {n: None for n in range(10)}
    for i, r in enumerate(data):
        for p in posisi:
            frek[nilai_pos(r, p)] += 1
        for n in {nilai_pos(r, p) for p in posisi}:
            terakhir[n] = i
    absen = jarak_absen(terakhir, total)
    n_obs = total * len(posisi)
    sd = math.sqrt(n_obs * 0.1 * 0.9)
    ambang = ambang_absen(1 - 0.9 ** len(posisi))
    hasil = []
    for n in range(10):
        z = round((frek[n] - n_obs * 0.1) / sd, 2) if sd else 0.0
        status = "HOT" if z >= Z_HOT else ("OVERDUE" if total and absen[n] >= ambang else "NORMAL")
        hasil.append({"angka": n, "frekuensi": frek[n],
                      "persen": round(frek[n] / n_obs * 100, 2) if n_obs else 0.0,
                      "z": z, "overdue": absen[n], "status": status})
    return hasil


def statistik_shio(data):
    total = len(data)
    frek = {no: 0 for no in SHIO_NAMA}
    terakhir = {no: None for no in SHIO_NAMA}
    for i, r in enumerate(data):
        frek[r["shio_no"]] += 1
        terakhir[r["shio_no"]] = i
    absen = jarak_absen(terakhir, total)
    hasil = []
    for s in SHIO_LIST:
        no = s["no"]
        p = SHIO_PROB[no]
        sd = math.sqrt(total * p * (1 - p))
        z = round((frek[no] - total * p) / sd, 2) if sd else 0.0
        status = "HOT" if z >= Z_HOT else ("OVERDUE" if total and absen[no] >= ambang_absen(p) else "NORMAL")
        hasil.append({"shio": SHIO_NAMA[no], "jumlah": frek[no],
                      "persen": round(frek[no] / total * 100, 2) if total else 0.0,
                      "harapan": round(p * 100, 2), "z": z, "overdue": absen[no], "status": status})
    return hasil


def _kls_status(status):
    return {"HOT": "r-hot", "OVERDUE": "r-overdue"}.get(status, "")


def tabel_digit(rows):
    urut = sorted(rows, key=lambda a: (-a["frekuensi"], a["angka"]))
    return tbl(["Angka", "Frek", "Persen", "Z", "Absen", "Status"],
               [(a["angka"], a["frekuensi"], f'{a["persen"]}%', a["z"], a["overdue"], badge(a["status"]))
                for a in urut],
               baris_kls=[_kls_status(a["status"]) for a in urut])


def tabel_shio(rows):
    urut = sorted(rows, key=lambda s: (-s["jumlah"], s["shio"]))
    return tbl(["Shio", "Jumlah", "Persen", "Harapan", "Z", "Absen", "Status"],
               [(s["shio"], s["jumlah"], f'{s["persen"]}%', f'{s["harapan"]}%', s["z"], s["overdue"],
                 badge(s["status"])) for s in urut],
               baris_kls=[_kls_status(s["status"]) for s in urut])


def baris_ringkas(label, isi_chips):
    return f'<div class="rrow"><span class="rlabel">{label}</span><div class="rchips">{isi_chips}</div></div>'


def ringkasan_pred(vp, step, stat):
    """Ringkasan per posisi & per pasangan, dalam bentuk kartu dengan chip ber-highlight."""
    kartu = []
    for nama, st in stat.items():
        hot = sorted([a for a in st if a["status"] == "HOT"],
                     key=lambda a: (-a["frekuensi"], a["angka"]))[:JUMLAH_ANGKA_HOT]
        ovd = sorted(st, key=lambda a: (-a["overdue"], a["angka"]))[:TOP_OVERDUE_RINGKAS]
        hot_set = {a["angka"] for a in hot}
        ovd_set = {a["angka"] for a in ovd}
        sama = hot_set & ovd_set
        c_hot = "".join(chip_info(a["angka"], f'{a["frekuensi"]}× · z {a["z"]}', "hot", a["angka"] in sama,
                                  f'Frekuensi {a["frekuensi"]} kali, z={a["z"]}') for a in hot) or '<span class="kosong">—</span>'
        c_ovd = "".join(chip_info(a["angka"], f'absen {a["overdue"]}', "ovd", a["angka"] in sama,
                                  "Melewati ambang overdue" if a["status"] == "OVERDUE" else "Absen terlama")
                        for a in ovd)
        c_sama = "".join(chip_info(n, "HOT + absen lama", "sama", True) for n in sorted(sama)) \
            or '<span class="kosong">tidak ada</span>'
        kartu.append(card(nama,
                          baris_ringkas("🔥 HOT", c_hot)
                          + baris_ringkas("⏳ OVERDUE", c_ovd)
                          + baris_ringkas("✅ Irisan", c_sama),
                          "ringkas"))
    for pk, v in vp.items():
        if not v:
            continue
        lbl = PAIRS[pk]["label"]
        kc = Counter(r["kode"] for r in v)
        c_pas = "".join(chip_info(k, f"{c}×", "hot", False, f"{SHIO_NAMA[KODE2_SHIO[k]]} — {c} kali")
                        for k, c in sorted(kc.items(), key=lambda kv: (-kv[1], kv[0]))[:JUMLAH_PASANGAN_HOT])
        isi = baris_ringkas("🔥 Pasangan HOT", c_pas)
        if len(v) >= 2:
            akhir = v[-1]["kode"]
            c = Counter(b["kode"] for a, b in zip(v, v[1:])
                        if a["kode"] == akhir and b["slot"] - a["slot"] == step)
            c_pola = "".join(chip_info(f"{akhir}→{k}", f"{n}×", "pred")
                             for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:JUMLAH_POLA_URUTAN]) \
                or '<span class="kosong">belum ada riwayat</span>'
            isi += baris_ringkas("🎯 Pola urutan", c_pola)
        kartu.append(card(lbl, isi, "ringkas pasangan"))
    return grid(*kartu)


def ringkasan_gabungan(stats):
    """Konsensus tiap posisi di tab D, E, dan Gabungan.
    Angka yang HOT / OVERDUE di minimal 2 dari 3 tab diberi highlight."""
    nama_tab = {"D": "☀️ D", "E": "🌙 E", "G": "Gabungan"}
    urutan_pos = []
    for st in stats.values():
        for nama in st:
            if nama not in urutan_pos:
                urutan_pos.append(nama)
    rows = []
    for nama in urutan_pos:
        ada = {t: st[nama] for t, st in stats.items() if nama in st}
        if not ada:
            continue
        ambang = min(2, len(ada))
        hot_t = {t: {a["angka"] for a in s if a["status"] == "HOT"} for t, s in ada.items()}
        ovd_t = {t: {a["angka"] for a in s if a["status"] == "OVERDUE"} for t, s in ada.items()}
        c_hot = Counter(x for s in hot_t.values() for x in s)
        c_ovd = Counter(x for s in ovd_t.values() for x in s)
        kons_hot = {x for x, c in c_hot.items() if c >= ambang}
        kons_ovd = {x for x, c in c_ovd.items() if c >= ambang}
        sel_tab = []
        for t in ("D", "E", "G"):
            if t not in ada:
                sel_tab.append(Raw('<span class="kosong">—</span>'))
                continue
            isi = "".join(chip_info(x, "", "hot", x in kons_hot) for x in sorted(hot_t[t]))
            isi += "".join(chip_info(x, "", "ovd", x in kons_ovd) for x in sorted(ovd_t[t]))
            sel_tab.append(Raw(isi or '<span class="kosong">—</span>'))
        k_hot = Raw("".join(chip_info(x, f"{c_hot[x]}/{len(ada)}", "sama", True) for x in sorted(kons_hot))
                    or '<span class="kosong">—</span>')
        k_ovd = Raw("".join(chip_info(x, f"{c_ovd[x]}/{len(ada)}", "ovd", True) for x in sorted(kons_ovd))
                    or '<span class="kosong">—</span>')
        rows.append((Raw(f"<b>{e(nama)}</b>"), *sel_tab, k_hot, k_ovd))
    return tbl(["Posisi", nama_tab["D"], nama_tab["E"], nama_tab["G"],
                "⭐ Konsensus HOT", "⭐ Konsensus OVERDUE"], rows)


def bagian3(recs, views):
    def proses(label, cond, step):
        idx = [i for i, r in enumerate(recs) if cond(r)]
        data = [recs[i] for i in idx]
        if not data:
            return catatan_html("Tidak ada data"), {}
        idx_set = set(idx)
        data_fb = [r for r in data if r["fb"] is not None]
        vp = {pk: [v for v in views[pk] if v["idx"] in idx_set] for pk in PAIRS}

        stat = {"Semua posisi": statistik_digit(data, [0, 1, 2, 3])}
        for p in range(4):
            stat[f"Ball {p + 1}"] = statistik_digit(data, [p])
        if data_fb:
            stat["Fireball"] = statistik_digit(data_fb, ["fb"])

        ho_cards = [ho_card(f"Ball {p + 1}", *hot_overdue([r["b"][p] for r in data], range(10), TOP_DIGIT))
                    for p in range(4)]
        if data_fb:
            ho_cards.append(ho_card("Fireball",
                                    *hot_overdue([r["fb"] for r in data_fb], range(10), TOP_DIGIT)))

        shio_cards = [card(f"Shio — {PAIRS[pk]['label']}", tabel_shio(statistik_shio(vp[pk])))
                      for pk in PAIRS if vp[pk]]

        out = [f'<p class="meta">{label}: {len(data)} undian diproses '
               f'(Fireball valid: {len(data_fb)})</p>',
               "<h4>🔮 Ringkasan per posisi</h4>", legenda(), ringkasan_pred(vp, step, stat),
               "<h4>🔢 Angka</h4>",
               grid(*[card(nama, tabel_digit(st)) for nama, st in stat.items()]),
               "<h4>🔥⏳ HOT &amp; OVERDUE per Ball &amp; Fireball</h4>",
               grid(*ho_cards),
               "<h4>🐯 Shio</h4>",
               grid(*shio_cards)]
        return "".join(out), stat

    items, stats = [], {}
    for nama_tab, kunci, label, cond, step in [
        ("☀️ Siang (D)", "D", "SIANG (D)", lambda r: r["tipe"] == "D", 2),
        ("🌙 Malam (E)", "E", "MALAM (E)", lambda r: r["tipe"] == "E", 2),
        ("Gabungan", "G", "GABUNGAN", lambda r: True, 1),
    ]:
        h, st = proses(label, cond, step)
        items.append((nama_tab, h))
        if st:
            stats[kunci] = st

    gab = ""
    if stats:
        gab = ("<h4>⭐ Ringkasan gabungan tiap posisi</h4>"
               + legenda()
               + card("Konsensus Siang · Malam · Gabungan", ringkasan_gabungan(stats), "gabungan")
               + catatan_html("Chip bercahaya = angka yang sama-sama HOT (merah) atau OVERDUE (kuning) "
                              "di minimal 2 dari 3 tab. Kolom konsensus menampilkan berapa tab yang sepakat (mis. 3/3)."))
    note = catatan_html(f"HOT = frekuensi di atas harapan (z ≥ {Z_HOT}). OVERDUE = absen cukup lama "
                        f"sehingga peluang kebetulannya ≤ {int(P_OVERDUE * 100)}%. "
                        f"Fireball dihitung hanya dari undian dengan Fireball valid.")
    return gab + tabs("b3", items, "sub") + note


# ==================================================
# BAGIAN 4 — HISTORI BALL 1-4 (pertama muncul → terbaru)
# ==================================================
def bagian4(recs):
    n = len(recs)
    if n == 0:
        return catatan_html("Tidak ada data")

    def hasil(i, p=None):
        """Hasil Ball 1-4 (+ Fireball). Digit di posisi p ditebalkan bila html=True."""
        b = recs[i]["b"]
        return "".join(f"<b>{x}</b>" if k == p else str(x) for k, x in enumerate(b))

    def lbl(i, p):
        fb = recs[i]["fb"]
        return Raw(f'{e(recs[i]["tanggal"])} ({e(recs[i]["tipe"])}) : {hasil(i, p)}'
                   f' <span class="meta">FB:{fb if fb is not None else "-"}</span>')

    def lbl_teks(i):
        fb = recs[i]["fb"]
        return (f'{recs[i]["tanggal"]} ({recs[i]["tipe"]}) : {"".join(map(str, recs[i]["b"]))}'
                f' FB:{fb if fb is not None else "-"}')

    harapan = n * 0.1
    sd = math.sqrt(n * 0.1 * 0.9)
    contoh = None   # baris Ball 1 angka 0, dipakai untuk contoh di panduan

    items = []
    for p in range(4):
        rows, detail = [], []
        for d in range(10):
            pos = [i for i, r in enumerate(recs) if r["b"][p] == d]
            z = round((len(pos) - harapan) / sd, 2) if sd else 0.0
            if not pos:
                rows.append((d, "-", "-", 0, "-", n, round(harapan, 1), z))
                continue
            # jeda = jumlah undian yang tidak memunculkan angka d di antara dua kemunculan
            gaps = [(b - a - 1, a, b) for a, b in zip(pos, pos[1:])]
            if gaps:
                g, a, b = max(gaps, key=lambda x: x[0])
                terlama = f"{g} undian ({recs[a]['tanggal']} → {recs[b]['tanggal']})"
            else:
                g, terlama = None, "-"
            sekarang = n - 1 - pos[-1]
            rows.append((d, lbl(pos[0], p), lbl(pos[-1], p), len(pos), terlama, sekarang, round(harapan, 1), z))
            if p == 0 and d == 0:
                contoh = (lbl_teks(pos[0]), lbl_teks(pos[-1]), len(pos), g, terlama, sekarang, z)

            # daftar semua kemunculan, lama → baru
            iso = [(k + 1, recs[i]["tanggal"], recs[i]["tipe"], Raw(hasil(i, p)),
                    recs[i]["fb"] if recs[i]["fb"] is not None else "-",
                    (i - pos[k - 1] - 1) if k else "-")
                   for k, i in enumerate(pos)]
            detail.append((len(pos), d, Raw(f'<details><summary>Angka {d} — {len(pos)} kali muncul</summary>'
                                            f'{tbl(["No", "Tanggal", "Sesi", "Hasil Ball 1-4", "FB", "Jeda dari sebelumnya (undian)"], iso)}'
                                            f'</details>')))

        rows.sort(key=lambda r: (r[3], r[0]))          # Total: terendah → tertinggi
        detail.sort(key=lambda x: (x[0], x[1]))
        kls_z = ["r-hot" if r[7] >= Z_HOT else ("r-overdue" if r[7] <= -Z_HOT else "") for r in rows]
        isi = tbl(["Angka", "Pertama muncul", "Terakhir muncul", "Total", "Jeda terlama",
                   "Absen sekarang", "Harapan", "Z"], rows, baris_kls=kls_z)
        isi += "".join(x[2] for x in detail)
        items.append((f"Ball {p + 1}", isi))

    # semua data, terbaru → terlama, angka tampil sebagai bola
    def bola(x, fb=False):
        return Raw(f'<span class="bola{" fb" if fb else ""}">{e(x)}</span>')

    semua = [(i + 1, r["tanggal"], r["tipe"],
              *[bola(x) for x in r["b"]],
              bola(r["fb"], True) if r["fb"] is not None else "-")
             for i, r in reversed(list(enumerate(recs)))]
    items.append(("Semua data",
                  tbl(["No", "Tanggal", "Sesi", "Ball 1", "Ball 2", "Ball 3", "Ball 4", "Fireball"], semua,
                      tinggi=True)))

    # panduan membaca (memakai angka nyata dari datamu)
    panduan_rows = [
        ("Angka", "Angka 0-9 pada Ball yang sedang dibuka (tab Ball 1-4)."),
        ("Pertama muncul", "Tanggal, sesi, dan hasil Ball 1-4 saat angka itu pertama kali keluar "
                           "(angka yang dibahas ditebalkan; FB = Fireball)."),
        ("Terakhir muncul", "Tanggal, sesi, dan hasil Ball 1-4 saat angka itu terakhir keluar."),
        ("Total", "Berapa kali angka itu keluar selama periode data."),
        ("Jeda terlama", "Rekor absen terpanjang: jumlah undian berturut-turut tanpa angka itu, "
                         "beserta rentang tanggal sebelum & sesudahnya."),
        ("Absen sekarang", "Jumlah undian terakhir tanpa angka itu. 0 = muncul di undian paling akhir."),
        ("Harapan", f"Berapa kali seharusnya muncul kalau murni kebetulan: {n} undian × 10% = {harapan:.1f}."),
        ("Z", "Selisih Total dari Harapan dalam satuan simpangan baku. "
              "Antara -2 dan +2 = wajar; di luar itu baru tergolong menonjol."),
    ]
    panduan = tbl(["Kolom", "Arti"], panduan_rows)
    if contoh:
        pertama, terakhir, total, g, terlama, sekarang, z = contoh
        panduan += catatan_html(
            f"Contoh (Ball 1, angka 0): pertama muncul {pertama}, terakhir {terakhir}; keluar {total} kali "
            f"(harapan {harapan:.1f}, Z = {z}). Jeda terlama: {terlama}. "
            f"Absen sekarang {sekarang}" + (" = muncul di undian paling akhir." if sekarang == 0 else " undian."))
    panduan_html = Raw(f'<details class="panduan"><summary>📖 Cara membaca tabel ini</summary>{panduan}</details>')

    note = catatan_html(f"Data dari {recs[0]['tanggal']} sampai {recs[-1]['tanggal']} ({n} undian). "
                        f"Jeda = jumlah undian (D dan E dihitung terpisah, sehari ada 2 undian) "
                        f"yang tidak memunculkan angka itu. Urutan angka: dari yang paling sedikit muncul ke yang paling sering. "
                        f"Tab 'Semua data': terbaru di atas.")
    return panduan_html + tabs("b4", items, "sub") + note


# ==================================================
# BAGIAN 5 — ANGKA BERUNTUN DALAM N DRAW TERBARU (Ball 1-4)
# ==================================================
def bagian5(recs, gid="b5"):
    m = min(JUMLAH_DRAW_BERUNTUN, len(recs))
    if m < 2:
        return catatan_html("Data kurang dari 2 undian — Bagian 5 dilewati.")
    last = recs[-m:]                      # urut lama → baru

    def runs(seq):
        """Daftar (angka, panjang, idx_awal, idx_akhir) untuk angka yang sama berturut-turut."""
        out, s0 = [], 0
        for i in range(1, len(seq) + 1):
            if i == len(seq) or seq[i] != seq[s0]:
                out.append((seq[s0], i - s0, s0, i - 1))
                s0 = i
        return out

    def bola(x, kls=""):
        return f'<span class="bola kecil {kls}">{e(x)}</span>'

    ringkas_rows, items = [], []
    for p in range(4):
        seq = [r["b"][p] for r in last]
        rn = runs(seq)
        ulang = [r for r in rn if r[1] >= 2]
        dalam_run = {i for r in ulang for i in range(r[2], r[3] + 1)}
        strip = "".join(bola(x, f"c{p + 1} " + ("ulang" if i in dalam_run else "")) for i, x in enumerate(seq))
        terpanjang = max(rn, key=lambda r: (r[1], r[3]))
        sekarang = rn[-1]
        chip_ulang = "".join(
            chip_info(d, f"{pj}× beruntun", f"cb{p + 1}", pj >= 3,
                      f"{last[a]['tanggal']} ({last[a]['tipe']}) → {last[b]['tanggal']} ({last[b]['tipe']})")
            for d, pj, a, b in ulang) or '<span class="kosong">tidak ada</span>'
        ringkas_rows.append((Raw(f'<span class="lb c{p + 1}">Ball {p + 1}</span>'), Raw(f'<span class="seq">{strip}</span>'),
                             Raw(chip_ulang),
                             f"{terpanjang[0]} × {terpanjang[1]}" if terpanjang[1] >= 2 else "tidak ada",
                             f"{sekarang[0]} × {sekarang[1]}"))

        rows, kls = [], []
        for d in range(10):
            idx = [i for i, x in enumerate(seq) if x == d]
            if not idx:
                continue
            panjang = max(r[1] for r in rn if r[0] == d)
            cur = sekarang[1] if sekarang[0] == d else 0
            jejak = Raw(f'<span class="strip c{p + 1}">' + "".join(
                f'<span class="dot{" on" if x == d else ""}"></span>' for x in seq) + "</span>")
            rows.append((d, len(idx), panjang, cur, jejak))
        rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
        kls = ["r-hot" if r[2] >= 2 else "" for r in rows]
        isi = (f'<p class="meta">Jejak dibaca dari kiri (paling lama) ke kanan (terbaru): '
               f'{e(last[0]["tanggal"])} ({e(last[0]["tipe"])}) → {e(last[-1]["tanggal"])} ({e(last[-1]["tipe"])})</p>'
               + tbl(["Angka", f"Muncul (dari {m} draw)", "Beruntun terpanjang", "Beruntun sekarang", "Jejak"],
                     rows, baris_kls=kls))
        items.append((f"Ball {p + 1}", isi))

    # draw terbaru → terlama; bola yang sama dengan draw sebelumnya diberi lingkaran
    draw_rows = []
    for i in range(m - 1, -1, -1):
        r = last[i]
        sel_b = [Raw(bola(x, f"c{k + 1} " + ("ulang" if i > 0 and x == last[i - 1]["b"][k] else "")))
                 for k, x in enumerate(r["b"])]
        fbv = Raw(bola(r["fb"], "fb")) if r["fb"] is not None else "-"
        draw_rows.append((i + 1, r["tanggal"], r["tipe"], *sel_b, fbv))
    items.append((f"{m} draw terbaru",
                  tbl(["No", "Tanggal", "Sesi", "Ball 1", "Ball 2", "Ball 3", "Ball 4", "Fireball"], draw_rows)))

    def tabel_sifat(judul, nama_a, nama_b, kode_a, kode_b, fungsi_a):
        """Kemunculan dua kelompok (mis. Ganjil/Genap) per Ball pada m draw terbaru."""
        rows_s = []
        for p in range(4):
            seq = [r["b"][p] for r in last]
            lab = [nama_a if fungsi_a(x) else nama_b for x in seq]
            na = lab.count(nama_a)
            nb = m - na
            rn = runs(lab)
            tp = max(rn, key=lambda r: (r[1], r[3]))
            sk = rn[-1]
            ulang = [r for r in rn if r[1] >= 2]
            dalam = {i for r in ulang for i in range(r[2], r[3] + 1)}
            strip = "".join(f'<span class="pf {"a" if l == nama_a else "b"}{" ulang" if i in dalam else ""}" title="{x}">'
                            f'{kode_a if l == nama_a else kode_b}</span>' for i, (l, x) in enumerate(zip(lab, seq)))
            chip_run = "".join(
                chip_info(l, f"{pj}× beruntun", "ga" if l == nama_a else "gb", pj >= 3,
                          f"{last[a]['tanggal']} ({last[a]['tipe']}) → {last[b]['tanggal']} ({last[b]['tipe']})")
                for l, pj, a, b in ulang) or '<span class="kosong">tidak ada</span>'

            def sel_n(n):
                return Raw(chip_info(n, f"{n / m * 100:.0f}%", "hot" if n / m >= BATAS_DOMINAN else ""))
            rows_s.append((Raw(f'<span class="lb c{p + 1}">Ball {p + 1}</span>'),
                           Raw(f'<span class="seq">{strip}</span>'), sel_n(na), sel_n(nb),
                           f"{tp[0]} × {tp[1]}", f"{sk[0]} × {sk[1]}", Raw(chip_run)))
        return card(judul, tbl(["Posisi", "Urutan (lama → baru)", nama_a, nama_b,
                                "Beruntun terpanjang", "Sedang berjalan", "Rangkaian beruntun (≥2)"], rows_s,
                               baris_kls=["rb1", "rb2", "rb3", "rb4"]))

    sifat = ("<h4>⚖️ Ganjil/Genap &amp; Besar/Kecil</h4>"
             + grid(tabel_sifat(f"Ganjil / Genap — {m} draw terbaru", "Ganjil", "Genap", "Gj", "Gn",
                                lambda x: x % 2 == 1),
                    tabel_sifat(f"Besar / Kecil — {m} draw terbaru", "Besar", "Kecil", "B", "K",
                                lambda x: x >= 5))
             + catatan_html(f"Ganjil = 1,3,5,7,9 · Genap = 0,2,4,6,8 · Besar = 5-9 · Kecil = 0-4. "
                            f"Kotak gelap = Ganjil/Besar, kotak berbingkai = Genap/Kecil (arahkan kursor untuk melihat angkanya). "
                            f"Kotak berlingkar merah dan chip 'beruntun' = dua draw atau lebih berturut-turut di sisi yang sama (arahkan kursor ke chip untuk melihat tanggalnya). "
                            f"Jumlah merah = salah satu sisi mencapai {int(BATAS_DOMINAN * 100)}% atau lebih dari {m} draw "
                            f"(harapan kebetulan: 50/50)."))

    ringkas = card(f"Beruntun dalam {m} draw terbaru",
                   tbl(["Posisi", "Urutan (lama → baru)", "Angka beruntun (≥2)", "Terpanjang", "Sedang berjalan"],
                       ringkas_rows, baris_kls=["rb1", "rb2", "rb3", "rb4"]), "gabungan")
    note = catatan_html(f"Beruntun = angka yang sama keluar di dua draw atau lebih secara berturut-turut pada Ball yang sama. "
                        f"Draw D dan E digabung menurut waktu (sehari 2 draw). Bola berlingkar merah = bagian dari rangkaian beruntun. "
                        f"'Sedang berjalan' = rangkaian yang berakhir di draw paling baru (× 1 berarti belum beruntun).")
    return "<h4>🔁 Ringkasan beruntun</h4>" + ringkas + sifat + "<h4>📋 Rincian per Ball</h4>" + tabs(gid, items, "sub") + note


# ==================================================
# HALAMAN HTML
# ==================================================
CSS = """
:root{
--bg:#f2f4f7;--fg:#141a26;--mut:#5f6b7d;--card:#ffffff;--bd:#dde2ea;--row:#f7f9fc;--ac:#2556d9;--ac-soft:#e4ebff;
--hero1:#101a33;--hero2:#1d3a8a;
--hot:#fde1de;--hotf:#a8200f;--hot-soft:rgba(224,60,40,.09);--hot-glow:rgba(224,60,40,.45);
--ovd:#fff0cc;--ovdf:#8a5200;--ovd-soft:rgba(240,160,0,.12);--ovd-glow:rgba(240,160,0,.55);
--sama:#d6f5df;--samaf:#14692f;--sama-glow:rgba(30,160,80,.5);
--pred:#e1eaff;--predf:#1c48b0;
--sh:0 1px 2px rgba(16,26,51,.06),0 6px 18px rgba(16,26,51,.05);--r:12px}
@media(prefers-color-scheme:dark){:root{
--bg:#0d1118;--fg:#e8ecf3;--mut:#93a0b4;--card:#161c26;--bd:#2a3342;--row:#1a212d;--ac:#7aa5ff;--ac-soft:#1c2b4d;
--hero1:#0b1226;--hero2:#16306e;
--hot:#4b1d19;--hotf:#ffb7ad;--hot-soft:rgba(255,110,90,.10);--hot-glow:rgba(255,110,90,.55);
--ovd:#4a3410;--ovdf:#ffd27f;--ovd-soft:rgba(255,190,70,.11);--ovd-glow:rgba(255,190,70,.55);
--sama:#173a22;--samaf:#86e0a0;--sama-glow:rgba(90,210,130,.55);
--pred:#1b2c52;--predf:#abc6ff;
--sh:0 1px 2px rgba(0,0,0,.4),0 6px 18px rgba(0,0,0,.3)}}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 "Inter","Segoe UI",system-ui,-apple-system,Roboto,sans-serif;font-variant-numeric:tabular-nums}
.hero{background:linear-gradient(135deg,var(--hero1),var(--hero2));color:#fff;padding:26px 16px 20px}
.wrap{max-width:1320px;margin:auto}
.hero h1{margin:0 0 10px;font-size:26px;font-weight:750;letter-spacing:-.01em}
.pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:999px;padding:3px 12px;font-size:12.5px;
max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill em{font-style:normal;opacity:.7;margin-right:6px}
.hero .note{color:#dbe4ff;border-color:rgba(255,255,255,.35);margin:10px 0 0}
main{max-width:1320px;margin:auto;padding:0 16px 56px}
h3{margin:0 0 10px;font-size:15px;font-weight:700}
h4{margin:26px 0 10px;font-size:16px;font-weight:700;display:flex;align-items:center;gap:8px}
h4::after{content:"";flex:1;height:1px;background:var(--bd)}
/* tab */
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0}
.tabs button{border:1px solid var(--bd);background:var(--card);color:var(--fg);padding:7px 15px;border-radius:999px;
cursor:pointer;font:inherit;font-weight:600;transition:background .15s,color .15s,border-color .15s}
.tabs button:hover{border-color:var(--ac);color:var(--ac)}
.tabs button.on{background:var(--ac);border-color:var(--ac);color:#fff;box-shadow:0 2px 8px var(--ac-soft)}
.tabs.top{position:sticky;top:0;z-index:40;margin:0 -16px 14px;padding:10px 16px;background:var(--bg);
border-bottom:1px solid var(--bd);backdrop-filter:blur(8px)}
.tabs.hari{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:8px;box-shadow:var(--sh)}
.tabs.hari button{padding:5px 13px;font-size:13px;border-radius:8px}
.tabs.sub button{padding:5px 12px;font-size:13px;border-radius:8px}
.panel{display:none}.panel.on{display:block}
/* kartu */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);padding:14px;min-width:0;
margin-bottom:14px;box-shadow:var(--sh)}
.grid .card{margin-bottom:0}
.card.acuan{border-color:var(--ac);box-shadow:0 0 0 2px var(--ac-soft),var(--sh)}
.card.ringkas{border-top:3px solid var(--ac)}
.card.ringkas.pasangan{border-top-color:var(--predf)}
.card.gabungan{border-top:3px solid var(--samaf)}
/* tabel */
.scroll{overflow-x:auto;border-radius:8px}
.scroll.tall{max-height:70vh;overflow-y:auto}
table{border-collapse:separate;border-spacing:0;width:100%;margin:6px 0}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--bd);white-space:nowrap}
td:has(.chip){white-space:normal}
tbody tr:nth-child(even) td{background:var(--row)}
tbody tr:hover td{background:var(--ac-soft)}
th{position:sticky;top:0;background:var(--card);color:var(--mut);font-weight:650;font-size:12.5px;z-index:1;
border-bottom:2px solid var(--bd)}
tr.r-hot td{background:var(--hot-soft)!important}
tr.r-hot td:first-child{box-shadow:inset 3px 0 0 var(--hotf)}
tr.r-overdue td{background:var(--ovd-soft)!important}
tr.r-overdue td:first-child{box-shadow:inset 3px 0 0 var(--ovdf)}
.meta{color:var(--mut);margin:6px 0;font-size:13px}
.note{color:var(--mut);font-size:13px;border-left:3px solid var(--ac);background:var(--card);
border-radius:0 8px 8px 0;padding:6px 12px;margin:12px 0}
.legenda{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;color:var(--mut);font-size:12.5px;margin:4px 0 10px}
/* chip */
.chip{display:inline-flex;align-items:baseline;gap:5px;min-width:24px;justify-content:center;padding:2px 9px;margin:2px 3px 2px 0;
border-radius:8px;font-weight:650;border:1px solid transparent}
.chip b{font-weight:750}
.chip small{font-size:11px;font-weight:500;opacity:.8}
.chip.hot{background:var(--hot);color:var(--hotf)}
.chip.ovd{background:var(--ovd);color:var(--ovdf)}
.chip.sama{background:var(--sama);color:var(--samaf)}
.chip.pred{background:var(--pred);color:var(--predf)}
.chip.kuat{box-shadow:0 0 0 2px var(--card),0 0 0 4px currentColor,0 0 12px var(--hot-glow);transform:translateY(-1px)}
.chip.ovd.kuat{box-shadow:0 0 0 2px var(--card),0 0 0 4px currentColor,0 0 12px var(--ovd-glow)}
.chip.sama.kuat{box-shadow:0 0 0 2px var(--card),0 0 0 4px currentColor,0 0 12px var(--sama-glow)}
.kosong{color:var(--mut)}
/* baris ringkasan */
.rrow{display:grid;grid-template-columns:96px 1fr;gap:8px;align-items:start;padding:7px 0;border-bottom:1px dashed var(--bd)}
.rrow:last-child{border-bottom:0}
.rlabel{font-weight:650;color:var(--mut);font-size:12.5px;padding-top:4px}
.rchips{display:flex;flex-wrap:wrap}
/* badge */
.badge{padding:1px 9px;border-radius:999px;font-size:12px;font-weight:650}
.badge.hot{background:var(--hot);color:var(--hotf)}
.badge.overdue{background:var(--ovd);color:var(--ovdf)}
.badge.normal{background:var(--bd);color:var(--mut)}
/* salin */
.kode{cursor:pointer}
.kode:hover{color:var(--ac);text-decoration:underline dotted}
.copy{margin-left:6px;padding:1px 9px;border:1px solid var(--ac);background:transparent;color:var(--ac);
border-radius:6px;font:inherit;font-size:12px;cursor:pointer}
.copy:hover{background:var(--ac);color:#fff}
.copy.ok{background:var(--samaf);border-color:var(--samaf);color:#fff}
/* bola */
.bola{display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:50%;
background:radial-gradient(circle at 32% 28%,#fff 0,#e4e7ec 45%,#a9b0ba 100%);color:#1c2128;font-weight:700;
border:1px solid #8f97a3;box-shadow:inset -2px -3px 5px rgba(0,0,0,.18),0 1px 3px rgba(0,0,0,.25)}
.bola.fb{background:radial-gradient(circle at 32% 28%,#ffd4cf 0,#f0594c 50%,#a8160a 100%);color:#fff;border-color:#8a1208}
/* warna per Ball: 1 biru, 2 hijau, 3 kuning, 4 emas */
:root{--c1:#4f74b0;--c2:#4c9468;--c3:#e0c460;--c4:#bd9140}
.bola.c1{background:radial-gradient(circle at 32% 28%,#c9d8ef 0,#4f74b0 60%,#37567f 100%);color:#fff;border-color:#37567f}
.bola.c2{background:radial-gradient(circle at 32% 28%,#c9e3d4 0,#4c9468 60%,#356c4b 100%);color:#fff;border-color:#356c4b}
.bola.c3{background:radial-gradient(circle at 32% 28%,#f6ecbf 0,#e0c460 60%,#b99d38 100%);color:#3f3300;border-color:#b99d38}
.bola.c4{background:radial-gradient(circle at 32% 28%,#eedab0 0,#bd9140 60%,#8a6624 100%);color:#2e1f00;border-color:#8a6624}
.lb{display:inline-block;min-width:64px;text-align:center;padding:2px 12px;border-radius:999px;font-weight:700}
.lb.c1{background:var(--c1);color:#fff}.lb.c2{background:var(--c2);color:#fff}
.lb.c3{background:var(--c3);color:#3f3300}.lb.c4{background:var(--c4);color:#2e1f00}
.chip.cb1{background:var(--c1);color:#fff}.chip.cb2{background:var(--c2);color:#fff}
.chip.cb3{background:var(--c3);color:#3f3300}.chip.cb4{background:var(--c4);color:#2e1f00}
.strip.c1 .dot.on{background:var(--c1)}.strip.c2 .dot.on{background:var(--c2)}
.strip.c3 .dot.on{background:var(--c3)}.strip.c4 .dot.on{background:var(--c4)}
tr.rb1 td:first-child{box-shadow:inset 4px 0 0 var(--c1)}tr.rb2 td:first-child{box-shadow:inset 4px 0 0 var(--c2)}
tr.rb3 td:first-child{box-shadow:inset 4px 0 0 var(--c3)}tr.rb4 td:first-child{box-shadow:inset 4px 0 0 var(--c4)}
.pf{display:inline-flex;justify-content:center;align-items:center;min-width:26px;height:22px;padding:0 3px;
border-radius:6px;font-size:11px;font-weight:700;margin-right:2px}
.pf.a{background:var(--fg);color:var(--bg)}
.pf.b{border:1px solid var(--mut);color:var(--mut)}
.pf.ulang{outline:2px solid var(--hotf);outline-offset:1px}
.chip.ga{background:var(--fg);color:var(--bg)}
.chip.gb{background:transparent;border-color:var(--mut);color:var(--fg)}
.bola.ulang{outline:2px solid var(--hotf);outline-offset:1px}
.seq{display:inline-flex;gap:3px;flex-wrap:nowrap}
.strip{display:inline-flex;gap:3px;align-items:center}
.dot{width:12px;height:12px;border-radius:3px;background:var(--bd)}
.dot.on{background:var(--hotf)}
.bola.kecil{width:22px;height:22px;font-size:12px;margin-right:2px;vertical-align:middle}
details{margin:8px 0}
summary{cursor:pointer;font-weight:650;padding:4px 0}
summary:hover{color:var(--ac)}
details.panduan{background:var(--card);border:1px solid var(--bd);border-radius:var(--r);padding:8px 14px;box-shadow:var(--sh)}
/* tombol gulir */
.nav-scroll{position:fixed;right:14px;bottom:18px;display:flex;flex-direction:column;gap:8px;z-index:50}
.nav-scroll button{width:42px;height:42px;border-radius:50%;border:0;background:var(--ac);color:#fff;
font-size:16px;line-height:1;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.3);opacity:.88}
.nav-scroll button:hover{opacity:1}
button:focus-visible,summary:focus-visible{outline:2px solid var(--ac);outline-offset:2px}
@media(max-width:640px){
.grid{grid-template-columns:1fr}
.hero h1{font-size:21px}
.rrow{grid-template-columns:1fr;gap:2px}
.tabs.top{margin:0 -16px 10px}}
@media print{.tabs,.nav-scroll{display:none}.panel{display:block!important}.card{box-shadow:none;break-inside:avoid}}
"""

JS = """document.addEventListener('click',function(ev){var b=ev.target.closest('button[data-t]');if(!b)return;
var g=b.dataset.g;document.querySelectorAll('[data-g="'+g+'"]').forEach(function(x){
x.classList.toggle('on',x===b||x.id===b.dataset.t);});});
document.addEventListener('click',function(ev){var n=ev.target.closest('button[data-scroll]');if(!n)return;
var tinggi=Math.max(document.documentElement.scrollHeight,document.body.scrollHeight);
window.scrollTo({top:n.dataset.scroll==='up'?0:tinggi,behavior:'smooth'});});
function salinFallback(t,ok){var a=document.createElement('textarea');a.value=t;a.style.position='fixed';a.style.opacity='0';
document.body.appendChild(a);a.select();var s=false;try{s=document.execCommand('copy');}catch(x){}a.remove();if(s)ok();}
function salin(t,ok){if(navigator.clipboard&&navigator.clipboard.writeText){
navigator.clipboard.writeText(t).then(ok,function(){salinFallback(t,ok);});}else{salinFallback(t,ok);}}
document.addEventListener('click',function(ev){var c=ev.target.closest('[data-copy]');if(!c)return;
var btn=c.tagName==='BUTTON'?c:c.parentNode.querySelector('button.copy');
salin(c.dataset.copy,function(){if(!btn)return;var o=btn.dataset.o||btn.textContent;btn.dataset.o=o;
btn.textContent='✓ Disalin';btn.classList.add('ok');
setTimeout(function(){btn.textContent=o;btn.classList.remove('ok');},1200);});});"""


def halaman(df, catatan, sumber, panel_items):
    n_d = int((df["Day/Eve"] == "D").sum())
    n_e = int((df["Day/Eve"] == "E").sum())

    def pill(k, v, judul=""):
        t = f' title="{e(judul)}"' if judul else ""
        return f'<span class="pill"{t}><em>{e(k)}</em>{e(v)}</span>'

    pills = "".join([
        pill("Sumber", sumber, sumber),
        pill("Data", f"{df['Date'].min():%Y-%m-%d} s/d {df['Date'].max():%Y-%m-%d}"),
        pill("Total", f"{len(df)} undian"),
        pill("☀️ D", n_d), pill("🌙 E", n_e),
        pill("Fireball valid", int(df['Fireball'].notna().sum())),
        pill("Dibuat", f"{datetime.now():%Y-%m-%d %H:%M:%S}"),
    ])
    notes = "".join(catatan_html(c) for c in catatan)
    disc = catatan_html("Undian acak & independen — semua hasil hanya merangkum sejarah, "
                        "bukan memprediksi undian berikutnya.")
    return (f'<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<meta name="color-scheme" content="light dark">'
            f'<title>Analisis Pick 4</title><style>{CSS}</style></head><body>'
            f'<header class="hero"><div class="wrap"><h1>Analisis Pick 4</h1>'
            f'<div class="pills">{pills}</div>{notes}</div></header>'
            f'<main>{tabs("top", panel_items, "top")}{disc}</main>'
            f'<div class="nav-scroll">'
            f'<button data-scroll="up" title="Ke atas" aria-label="Ke atas">▲</button>'
            f'<button data-scroll="down" title="Ke bawah" aria-label="Ke bawah">▼</button></div>'
            f'<script>{JS}</script></body></html>')


def buka_di_chrome(path):
    """Buka file HTML langsung di Google Chrome; jika Chrome tidak ketemu, pakai browser default."""
    path = os.path.abspath(path)
    kandidat = [
        os.path.join(os.environ.get("PROGRAMFILES", ""), r"Google\Chrome\Application\chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), r"Google\Chrome\Application\chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium",
    ]
    for exe in kandidat:
        if exe and os.path.isfile(exe):
            try:
                subprocess.Popen([exe, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                pass
    webbrowser.open("file:///" + path.replace("\\", "/").lstrip("/"))


def per_hari(kunci, df, recs, bangun):
    """Bungkus satu bagian dengan menu hari: 'Semua hari' + Senin..Minggu.
    bangun(df_sub, recs_sub, views_sub) -> HTML bagian itu untuk subset data tersebut."""
    global PREFIX
    pilihan = [("Semua hari", None)] + [(h, i) for i, h in enumerate(HARI_NAMA)]
    items = []
    for k, (lbl, hi) in enumerate(pilihan):
        if hi is None:
            df_s, recs_s = df, recs
        else:
            df_s = df[df["Date"].dt.weekday == hi].reset_index(drop=True)
            recs_s = [r for r in recs if r["hari"] == hi]
        if not recs_s:
            isi = catatan_html(f"Tidak ada data untuk {lbl}.")
        else:
            PREFIX = f"{kunci}h{k}-"
            try:
                views_s = {pk: buat_view(recs_s, pk) for pk in PAIRS}
                isi = bangun(df_s, recs_s, views_s)
            finally:
                PREFIX = ""
            if hi is not None and kunci in ("b1", "b2", "b3"):
                isi = catatan_html(f"Hanya memakai draw hari {lbl}: urutan antar-undian (transisi/pola) dihitung "
                                   f"di antara draw hari {lbl} saja.") + isi
        items.append((f"{lbl} ({len(recs_s)})", isi))
    return tabs(f"{kunci}hari", items, "hari")


# ==================================================
# MAIN
# ==================================================
def main():
    ap = argparse.ArgumentParser(description="Analisis Pick 4 — output satu file HTML")
    ap.add_argument("csv", nargs="?", default=None, help="path/URL CSV (default: file lokal, lalu URL GitHub)")
    ap.add_argument("--html", default=None, help="path file HTML hasil")
    ap.add_argument("--bagian", default="1,2,3,4,5", help="bagian yang dijalankan, mis. 1,3 (default: 1,2,3,4,5)")
    ap.add_argument("--tambahan", default="", help="pasangan tambahan untuk Bagian 2, mis. 12,47")
    ap.add_argument("--mulai", default=TANGGAL_MULAI,
                    help=f"tanggal awal YYYY-MM-DD (default: {TANGGAL_MULAI or 'semua data'}; "
                         f"\"\" = semua data)")
    ap.add_argument("--hari", default=HARI,
                    help="filter hari untuk semua bagian: semua, senin, selasa, rabu, kamis, jumat, sabtu, minggu "
                         f"(default: {HARI})")
    ap.add_argument("--tanpa-buka", action="store_true", help="jangan buka hasil di browser")
    args = ap.parse_args()

    bagian = {b.strip() for b in args.bagian.split(",") if b.strip()}
    if not bagian <= {"1", "2", "3", "4", "5"}:
        sys.exit("--bagian hanya boleh berisi 1, 2, 3, 4, 5")

    sumber = args.csv or (CSV_LOKAL if os.path.exists(CSV_LOKAL) else CSV_URL)
    folder = FOLDER_DEFAULT if os.path.isdir(FOLDER_DEFAULT) else os.getcwd()
    html_path = args.html or os.path.join(folder, NAMA_HTML)
    tambahan = [p.strip() for p in args.tambahan.split(",") if p.strip()]

    df, catatan = muat_data(sumber, args.mulai)
    hari = args.hari.strip().lower().replace("'", "")
    if hari not in ("", "semua"):
        peta = {h.lower(): i for i, h in enumerate(HARI_NAMA)}
        if hari not in peta:
            sys.exit(f"--hari tidak dikenal: '{args.hari}'. Pilih: semua, " + ", ".join(h.lower() for h in HARI_NAMA))
        df = df[df["Date"].dt.weekday == peta[hari]].reset_index(drop=True)
        if df.empty:
            sys.exit(f"Tidak ada data untuk hari {HARI_NAMA[peta[hari]]} pada rentang tanggal ini.")
        catatan.append(f"Filter hari aktif: hanya {HARI_NAMA[peta[hari]]} ({len(df)} undian). "
                       f"Urutan antar-undian di Bagian 1-3 jadi per-hari, bukan harian berurutan.")
    recs = buat_recs(df)

    panels = []
    if "1" in bagian:
        panels.append(("Bagian 1 — Fireball→Ball",
                       per_hari("b1", df, recs, lambda d, r, v: bagian1(d))))
    if "2" in bagian:
        panels.append(("Bagian 2 — Pola pasangan & shio",
                       per_hari("b2", df, recs, lambda d, r, v: tabs(
                           "b2", [(P["label"], bagian2(v[pk], pk, tambahan)) for pk, P in PAIRS.items()], "sub"))))
    if "3" in bagian:
        panels.append(("Bagian 3 — Statistik",
                       per_hari("b3", df, recs, lambda d, r, v: bagian3(r, v))))
    if "4" in bagian:
        panels.append(("Bagian 4 — Histori Ball 1-4",
                       per_hari("b4", df, recs, lambda d, r, v: bagian4(r))))
    if "5" in bagian:
        panels.append(("Bagian 5 — Beruntun 10 draw",
                       per_hari("b5", df, recs, lambda d, r, v: bagian5(r))))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(halaman(df, catatan, sumber, panels))
    if not args.tanpa_buka:
        buka_di_chrome(html_path)


if __name__ == "__main__":
    main()
