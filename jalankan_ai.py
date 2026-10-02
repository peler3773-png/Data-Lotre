# jalankan_ai.py — LSTM + ORDE 1 SAJA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED_TETAP = 20261003
import random
random.seed(SEED_TETAP)
import numpy as np
np.random.seed(SEED_TETAP)
import tensorflow as tf
tf.random.set_seed(SEED_TETAP)
tf.get_logger().setLevel('ERROR')

import urllib.request, json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10
LIMIT_DATA = 16250

# === ORDE 1 SAJA ===
class Orde1Model:
    def __init__(self):
        self.transisi = {}  # {angka_terakhir: {angka_selanjutnya: jumlah}}

    def latih(self, daftar_deret):
        self.transisi = {}
        semua = []
        for angka_4d in daftar_deret:
            semua.extend(angka_4d)
        for i in range(len(semua) - 1):
            a, b = semua[i], semua[i+1]
            self.transisi.setdefault(a, {})[b] = self.transisi[a].get(b, 0) + 1

    def prediksi(self, angka_terakhir):
        if angka_terakhir not in self.transisi:
            return np.ones(10) / 10  # rata kalau belum pernah
        t = self.transisi[angka_terakhir]
        total = sum(t.values())
        return np.array([t.get(d, 0) / total for d in range(10)])

def proses_semua():
    # === AMBIL DATA ===
    req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        isi_file = r.read().decode('utf-8')

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

    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[-LIMIT_DATA:]

    list_p = sorted(semua_pasaran)
    p2i = {p:i for i,p in enumerate(list_p)}
    n_p = len(list_p)

    # === LATIH ORDE 1 ===
    orde1 = Orde1Model()
    orde1.latih([d["angka"] for d in mentah_data])
    deret_penuh = []
    for d in mentah_data[-LOOKBACK:]: deret_penuh.extend(d["angka"])
    angka_terakhir = deret_penuh[-1]
    p_orde1 = orde1.prediksi(angka_terakhir)

    print("\n" + "="*55)
    print(f"📊 Total: {len(mentah_data)} baris | Pasaran: {n_p}")
    print(f"🔗 Orde 1 siap — Angka terakhir: {angka_terakhir}")
    print("="*55 + "\n")

    # === BANGUN LSTM ===
    total_sampel = len(mentah_data) - LOOKBACK
    if total_sampel < 1:
        print("⚠️ Data belum cukup!")
        return

    X = np.zeros((total_sampel, LOOKBACK, 4), dtype=np.float32)
    P = np.zeros(total_sampel, dtype=np.int32)
    Y = np.zeros((total_sampel, 4), dtype=np.int32)
    for i in range(total_sampel):
        X[i] = [mentah_data[j]["angka"] for j in range(i, i+LOOKBACK)]
        P[i] = p2i[mentah_data[i+LOOKBACK]["pasaran"]]
        Y[i] = mentah_data[i+LOOKBACK]["angka"]
    y_train = [Y[:,0], Y[:,1], Y[:,2], Y[:,3]]

    input_angka = Input(shape=(LOOKBACK,4))
    lstm = LSTM(64, activation='relu')(input_angka)
    input_pasaran = Input(shape=(1,))
    emb = Flatten()(Embedding(n_p, 8)(input_pasaran))
    gabung = concatenate([lstm, emb])
    hidden = Dense(32, activation='relu')(gabung)
    out = [Dense(10, activation='softmax')(hidden) for _ in range(4)]

    model = Model([input_angka, input_pasaran], out)
    model.compile('adam', loss='sparse_categorical_crossentropy')

    print("🧠 Melatih LSTM...")
    model.fit([X, P], y_train, epochs=50, batch_size=250, verbose=0)

    # === DATA TERAKHIR UNTUK PREDIKSI ===
    X_last = np.array([mentah_data[j]["angka"] for j in range(-LOOKBACK,0)], dtype=np.float32)
    X_last = np.expand_dims(X_last, 0)

    # === BOBOT GABUNGAN ===
    B_LSTM = 0.70   # LSTM lebih dominan
    B_O1   = 0.30   # Orde 1 pelengkap

    hasil = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB",
        "total_data": len(mentah_data),
        "jumlah_pasaran": n_p,
        "daftar_pasaran": list_p,
        "data_terbaru": data_terbaru,
        "seed": SEED_TETAP,
        "rumus": f"LSTM {B_LSTM*100:.0f}% + Orde1 {B_O1*100:.0f}%",
        "hasil": {}
    }

    nama_pos = ["AS","KOP","KEPALA","EKOR"]
    print("\n🎯 HASIL PREDIKSI — LSTM + ORDE 1")
    print("="*70)

    for p in list_p:
        pred_lstm = model.predict([X_last, np.array([p2i[p]])], verbose=0)
        dtr = data_terbaru[p]
        print(f"\n📌 {p:8} | Terakhir: {dtr['nomor']} | {dtr['tanggal']}")
        pos_data = {}
        for idx, nm in enumerate(nama_pos):
            # Gabungkan skor LSTM + Orde1
            skor = pred_lstm[idx][0] * B_LSTM + p_orde1 * B_O1
            urut = np.argsort(skor)[::-1]
            tujuh = list(urut[:6]) + [urut[9]]
            sembilan = list(urut[:9])
            pos_data[nm] = {
                "lima": [str(a) for a in tujuh],
                "sembilan": [str(a) for a in sembilan],
                "skor_tertinggi": f"{skor[urut[0]]:.4f}"
            }
            print(f"   {nm:6} | 7D: {''.join(pos_data[nm]['lima'])}  | 9D: {''.join(pos_data[nm]['sembilan'])}")
        hasil["hasil"][p] = pos_data

    with open("hasil_prediksi.json","w",encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print(f"✅ Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
