# jalankan_ai.py — VERSI TAHAN ERROR UNTUK GITHUB
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

SEED_TETAP = 20261002B
import random
random.seed(SEED_TETAP)
import numpy as np
np.random.seed(SEED_TETAP)

import urllib.request
import json
from datetime import datetime

# Coba impor TensorFlow — beri pesan jelas kalau gagal
try:
    import tensorflow as tf
    tf.random.set_seed(SEED_TETAP)
    tf.get_logger().setLevel('ERROR')
    TF_TERSEDIA = True
except Exception as e:
    print(f"⚠️ Gagal memuat TensorFlow: {e}")
    TF_TERSEDIA = False
    exit(1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10
LIMIT_DATA = 16250

# ==============================================
class MarkovModel:
    def __init__(self):
        self.orde1 = {}
        self.orde2 = {}

    def latih(self, daftar_angka_4d):
        self.orde1 = {}
        self.orde2 = {}
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
        trans = self.orde1[angka_terakhir]
        total = sum(trans.values())
        return np.array([trans.get(d,0)/total for d in range(10)])

    def prediksi_orde2(self, dua_terakhir):
        kunci = tuple(dua_terakhir)
        if kunci not in self.orde2:
            return np.ones(10) / 10
        trans = self.orde2[kunci]
        total = sum(trans.values())
        return np.array([trans.get(d,0)/total for d in range(10)])

def proses_semua():
    print(f"📥 Mengunduh data...")
    try:
        req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as r:
            isi_file = r.read().decode('utf-8')
    except Exception as e:
        print(f"❌ Gagal unduh data: {e}")
        return

    mentah_data = []
    data_terbaru_per_pasaran = {}
    semua_pasaran = set()

    for baris in isi_file.strip().splitlines():
        bagian = baris.split('|')
        if len(bagian) < 4: continue
        pasaran = bagian[0].strip().upper()
        tgl = bagian[1].strip()
        angka_4d = bagian[2].strip()
        waktu = bagian[3].strip()
        if len(angka_4d) == 4 and angka_4d.isdigit():
            arr = [int(d) for d in angka_4d]
            mentah_data.append({"pasaran":pasaran,"tanggal":tgl,"angka":arr,"nomor":angka_4d,"waktu":waktu})
            semua_pasaran.add(pasaran)
            data_terbaru_per_pasaran[pasaran] = {"nomor":angka_4d,"tanggal":tgl,"waktu":waktu}

    mentah_data.sort(key=lambda x:(x["tanggal"],x["waktu"]), reverse=True)
    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[:LIMIT_DATA]

    list_pasaran = sorted(semua_pasaran)
    pasaran_ke_idx = {p:i for i,p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)

    print("\n" + "="*55)
    print("📋 DATA TERBARU PER PASARAN")
    print("="*55)
    for p in list_pasaran:
        d = data_terbaru_per_pasaran[p]
        print(f" {p:8} | {d['nomor']:4} | {d['tanggal']} {d['waktu'][-8:]}")
    print("="*55)
    print(f"📊 Total Data: {len(mentah_data)} baris | Pasaran: {total_jenis_pasaran}")

    markov = MarkovModel()
    markov.latih([d["angka"] for d in mentah_data])
    print(f"🔗 Markov 1 & 2 → Siap")

    total_sampel = len(mentah_data) - LOOKBACK
    if total_sampel < 1:
        print("⚠️ Data belum cukup!")
        return

    X_angka = np.zeros((total_sampel, LOOKBACK, 4), dtype=np.float32)
    X_konteks = np.zeros(total_sampel, dtype=np.int32)
    Y = np.zeros((total_sampel, 4), dtype=np.int32)
    for i in range(total_sampel):
        X_angka[i] = [mentah_data[j]["angka"] for j in range(i, i+LOOKBACK)]
        X_konteks[i] = pasaran_ke_idx[mentah_data[i+LOOKBACK]["pasaran"]]
        Y[i] = mentah_data[i+LOOKBACK]["angka"]
    y_train = [Y[:,0], Y[:,1], Y[:,2], Y[:,3]]

    # Model lebih kecil untuk GitHub
    input_angka = Input(shape=(LOOKBACK,4), name='input_angka')
    lstm_layer = LSTM(32, activation='relu')(input_angka)
    input_pasaran = Input(shape=(1,), name='input_pasaran')
    emb = Flatten()(Embedding(total_jenis_pasaran, 4)(input_pasaran))
    gabung = concatenate([lstm_layer, emb])
    hidden = Dense(16, activation='relu')(gabung)
    out_as  = Dense(10, activation='softmax', name='as')(hidden)
    out_kop = Dense(10, activation='softmax', name='kop')(hidden)
    out_kep = Dense(10, activation='softmax', name='kep')(hidden)
    out_eko = Dense(10, activation='softmax', name='eko')(hidden)

    model = Model(inputs=[input_angka, input_pasaran], outputs=[out_as,out_kop,out_kep,out_eko])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    print(f"🧠 Melatih LSTM (50 Epoch / 250 Batch)...")
    model.fit({'input_angka':X_angka,'input_pasaran':X_konteks}, y_train,
              epochs=50, batch_size=250, verbose=0)

    input_terbaru = np.array([mentah_data[j]["angka"] for j in range(-LOOKBACK,0)], dtype=np.float32)
    input_terbaru = np.expand_dims(input_terbaru, 0)
    deret = []
    for d in mentah_data[-LOOKBACK:]: deret.extend(d["angka"])
    satu_terakhir = deret[-1]
    dua_terakhir = deret[-2:]

    B_LSTM, B_M1, B_M2 = 0.60, 0.25, 0.15
    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "total_data": len(mentah_data),
        "daftar_pasaran": list_pasaran,
        "data_terbaru": data_terbaru_per_pasaran,
        "seed": SEED_TETAP,
        "pengaturan": f"LIMIT={LIMIT_DATA} | EPOCH=50 | BATCH=250"
    }
    nama_pos = ["AS","KOP","KEPALA","EKOR"]
    hasil_akhir["hasil"] = {}

    print("\n🎯 PREDIKSI PER PASARAN")
    print("="*70)
    for p in list_pasaran:
        pred = model.predict([input_terbaru, np.array([pasaran_ke_idx[p]])], verbose=0)
        dtr = data_terbaru_per_pasaran[p]
        print(f"\n📌 {p:8} | Terakhir: {dtr['nomor']} | {dtr['tanggal']}")
        pos_data = {}
        for idx, nm in enumerate(nama_pos):
            sg = pred[idx][0]*B_LSTM + markov.prediksi_orde1(satu_terakhir)*B_M1 + markov.prediksi_orde2(dua_terakhir)*B_M2
            urut = np.argsort(sg)[::-1]
            tujuh = list(urut[:6]) + [urut[9]]
            pos_data[nm] = {
                "lima": [str(a) for a in tujuh],
                "sembilan": [str(a) for a in urut[:9]],
                "skor_tertinggi": f"{sg[urut[0]]:.4f}"
            }
            print(f"   {nm:6} | 7D: {''.join(pos_data[nm]['lima'])}  | 9D: {''.join(pos_data[nm]['sembilan'])}")
        hasil_akhir["hasil"][p] = pos_data

    with open("hasil_prediksi.json","w",encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print(f"✅ Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
