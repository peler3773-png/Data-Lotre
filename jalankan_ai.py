# jalankan_ai.py — OPTIMASI MEMORI: 65 Pasaran ✅
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

SEED_TETAP = "20261003A"
import hashlib
def seed_angka(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

import random
random.seed(seed_angka(SEED_TETAP))
import numpy as np
np.random.seed(seed_angka(SEED_TETAP))

import tensorflow as tf
tf.random.set_seed(seed_angka(SEED_TETAP))
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate

import urllib.request, json
from datetime import datetime

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10
LIMIT_DATA = 16250

class MarkovModel:
    def __init__(self):
        self.orde1 = {}
        self.orde2 = {}

    def latih(self, daftar_angka_4d):
        semua_deret = []
        for angka in daftar_angka_4d:
            semua_deret.extend(angka)
        for i in range(len(semua_deret) - 1):
            a, b = semua_deret[i], semua_deret[i+1]
            self.orde1.setdefault(a, {})[b] = self.orde1[a].get(b, 0) + 1
        for i in range(len(semua_deret) - 2):
            a, b, c = semua_deret[i], semua_deret[i+1], semua_deret[i+2]
            self.orde2.setdefault((a,b), {})[c] = self.orde2[(a,b)].get(c, 0) + 1

    def prediksi_orde1(self, angka_terakhir):
        if angka_terakhir not in self.orde1:
            return np.ones(10) / 10
        t = self.orde1[angka_terakhir]
        total = sum(t.values())
        return np.array([t.get(d,0)/total for d in range(10)])

    def prediksi_orde2(self, dua_terakhir):
        k = tuple(dua_terakhir)
        if k not in self.orde2:
            return np.ones(10) / 10
        t = self.orde2[k]
        total = sum(t.values())
        return np.array([t.get(d,0)/total for d in range(10)])

def proses_semua():
    print("📥 Mengunduh data...")
    try:
        req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            isi_file = r.read().decode('utf-8')
    except Exception as e:
        print(f"❌ Gagal unduh: {e}")
        return

    mentah_data = []
    data_terbaru = {}
    semua_pasaran = set()
    for baris in isi_file.strip().splitlines():
        b = baris.split('|')
        if len(b) < 4: continue
        p, tgl, no, wkt = b[0].strip().upper(), b[1].strip(), b[2].strip(), b[3].strip()
        if len(no) == 4 and no.isdigit():
            arr = [int(d) for d in no]
            mentah_data.append({"pasaran":p,"tanggal":tgl,"angka":arr,"nomor":no,"waktu":wkt})
            semua_pasaran.add(p)
            data_terbaru[p] = {"nomor":no,"tanggal":tgl,"waktu":wkt}

    mentah_data.sort(key=lambda x:(x["tanggal"],x["waktu"]), reverse=True)
    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[:LIMIT_DATA]

    list_p = sorted(semua_pasaran)
    p2i = {p:i for i,p in enumerate(list_p)}
    n_p = len(list_p)

    print("\n" + "="*55)
    print("📋 DATA TERBARU PER PASARAN")
    print("="*55)
    for p in list_p:
        d = data_terbaru[p]
        print(f" {p:8} | {d['nomor']:4} | {d['tanggal']} {d['waktu'][-8:]}")
    print(f"\n📊 Total: {len(mentah_data)} baris | Pasaran: {n_p}")

    markov = MarkovModel()
    markov.latih([d["angka"] for d in mentah_data])
    print("🔗 Markov Siap")

    n_sampel = len(mentah_data) - LOOKBACK
    if n_sampel < 1:
        print("⚠️ Data belum cukup!")
        return

    print("📦 Menyiapkan data latih...")
    X = np.zeros((n_sampel, LOOKBACK, 4), dtype=np.float32)
    P = np.zeros(n_sampel, dtype=np.int32)
    Y = np.zeros((n_sampel, 4), dtype=np.int32)
    for i in range(n_sampel):
        X[i] = [mentah_data[j]["angka"] for j in range(i, i+LOOKBACK)]
        P[i] = p2i[mentah_data[i+LOOKBACK]["pasaran"]]
        Y[i] = mentah_data[i+LOOKBACK]["angka"]
    yt = [Y[:,0], Y[:,1], Y[:,2], Y[:,3]]

    # ==============================================
    # MODEL SANGAT RINGKAS — untuk 65 pasaran di GitHub
    # ==============================================
    print("🧠 Membuat model ringkas...")
    in1 = Input(shape=(LOOKBACK, 4), name='in_angka')
    l1 = LSTM(16, activation='relu', name='lstm')(in1)  # Kecil: 16 unit
    in2 = Input(shape=(1,), name='in_pasaran')
    emb = Flatten(name='flt')(Embedding(n_p, 2, name='emb')(in2))  # Sangat kecil: 2 dimensi
    gab = concatenate([l1, emb], name='gab')
    hid = Dense(8, activation='relu', name='hid')(gab)  # Kecil: 8 unit
    out_as  = Dense(10, activation='softmax', name='as')(hid)
    out_kop = Dense(10, activation='softmax', name='kop')(hid)
    out_kep = Dense(10, activation='softmax', name='kep')(hid)
    out_eko = Dense(10, activation='softmax', name='eko')(hid)

    model = Model([in1, in2], [out_as, out_kop, out_kep, out_eko])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    print(f"🧠 Melatih {n_sampel} sampel...")
    model.fit([X, P], yt, epochs=30, batch_size=512, verbose=1)  # Batch besar = cepat & hemat memori

    # Data terakhir untuk prediksi
    X_last = np.array([mentah_data[j]["angka"] for j in range(-LOOKBACK,0)], dtype=np.float32)
    X_last = np.expand_dims(X_last, 0)
    deret = []
    for d in mentah_data[-LOOKBACK:]: deret.extend(d["angka"])
    s1, s2 = deret[-1], deret[-2:]

    bL, bM1, bM2 = 0.6, 0.25, 0.15
    hasil = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB",
        "total_data": len(mentah_data),
        "jumlah_pasaran": n_p,
        "daftar_pasaran": list_p,
        "data_terbaru": data_terbaru,
        "seed": SEED_TETAP,
        "pengaturan": f"LIMIT={LIMIT_DATA} | EPOCH=30 | BATCH=512 | LSTM=16",
        "hasil": {}
    }

    print("\n🎯 HASIL PREDIKSI")
    print("="*70)
    nama = ["AS","KOP","KEPALA","EKOR"]
    for p in list_p:
        pred = model.predict([X_last, np.array([p2i[p]])], verbose=0)
        dtr = data_terbaru[p]
        print(f"\n📌 {p:8} | Terakhir: {dtr['nomor']} | {dtr['tanggal']}")
        pos_data = {}
        for idx, nm in enumerate(nama):
            sg = pred[idx][0]*bL + markov.prediksi_orde1(s1)*bM1 + markov.prediksi_orde2(s2)*bM2
            urut = np.argsort(sg)[::-1]
            tujuh = list(urut[:6]) + [urut[9]]
            pos_data[nm] = {
                "lima": [str(a) for a in tujuh],
                "sembilan": [str(a) for a in urut[:9]],
                "skor_tertinggi": f"{sg[urut[0]]:.4f}"
            }
            print(f"   {nm:6} | 7D: {''.join(pos_data[nm]['lima'])}  | 9D: {''.join(pos_data[nm]['sembilan'])}")
        hasil["hasil"][p] = pos_data

    with open("hasil_prediksi.json","w",encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print(f"✅ Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
