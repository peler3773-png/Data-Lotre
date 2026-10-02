# jalankan_ai.py — FINAL: LSTM + MARKOV 1/2 + DATA TERBARU TEPAT
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 🔒 KUNCI ACAK — HASIL STABIL
SEED_TETAP = 20261002
import random
random.seed(SEED_TETAP)
import numpy as np
np.random.seed(SEED_TETAP)
import tensorflow as tf
tf.random.set_seed(SEED_TETAP)
# =================================

import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate

tf.get_logger().setLevel('ERROR')

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10
LIMIT_DATA = 10000

# ==============================================
# MODUL MARKOV ORDE 1 & ORDE 2
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
            a = semua_deret[i]
            b = semua_deret[i+1]
            if a not in self.orde1:
                self.orde1[a] = {}
            self.orde1[a][b] = self.orde1[a].get(b, 0) + 1

        for i in range(len(semua_deret) - 2):
            a = semua_deret[i]
            b = semua_deret[i+1]
            c = semua_deret[i+2]
            kunci = (a, b)
            if kunci not in self.orde2:
                self.orde2[kunci] = {}
            self.orde2[kunci][c] = self.orde2[kunci].get(c, 0) + 1

    def prediksi_orde1(self, angka_terakhir):
        if angka_terakhir not in self.orde1:
            return np.ones(10) / 10
        transisi = self.orde1[angka_terakhir]
        total = sum(transisi.values())
        skor = np.zeros(10)
        for d in range(10):
            skor[d] = transisi.get(d, 0) / total
        return skor

    def prediksi_orde2(self, dua_terakhir):
        kunci = tuple(dua_terakhir)
        if kunci not in self.orde2:
            return np.ones(10) / 10
        transisi = self.orde2[kunci]
        total = sum(transisi.values())
        skor = np.zeros(10)
        for d in range(10):
            skor[d] = transisi.get(d, 0) / total
        return skor


def proses_semua():
    # === AMBIL DATA ===
    req = urllib.request.Request(
        DATA_UNDIAN_URL,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        isi_file = r.read().decode('utf-8')

    mentah_data = []
    semua_pasaran = set()
    data_terbaru_per_pasaran = {}

    for baris in isi_file.strip().splitlines():
        bagian = baris.split('|')
        if len(bagian) < 4:
            continue
        pasaran = bagian[0].strip().upper()
        tgl = bagian[1].strip()
        angka_4d = bagian[2].strip()
        waktu = bagian[3].strip()

        if len(angka_4d) == 4 and angka_4d.isdigit():
            arr = [int(d) for d in angka_4d]
            mentah_data.append({
                "pasaran": pasaran,
                "tanggal": tgl,
                "angka": arr,
                "nomor": angka_4d,
                "waktu": waktu
            })
            semua_pasaran.add(pasaran)
            # ✅ Timpa = baris paling bawah = paling baru
            data_terbaru_per_pasaran[pasaran] = {
                "nomor": angka_4d,
                "tanggal": tgl,
                "waktu": waktu
            }

    # ✅ Urutkan: paling baru di posisi pertama
    mentah_data.sort(key=lambda x: (x["tanggal"], x["waktu"]), reverse=True)

    # ✅ Ambil 80rb PALING BARU saja
    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[:LIMIT_DATA]

    list_pasaran = sorted(semua_pasaran)
    pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)

    # ==============================================
    # 📋 CETAK DATA TERBARU
    # ==============================================
    print("\n" + "="*55)
    print("📋 DATA TERBARU PER PASARAN")
    print("="*55)
    for p in list_pasaran:
        d = data_terbaru_per_pasaran[p]
        print(f" {p:8} | {d['nomor']:4} | {d['tanggal']} {d['waktu'][-8:]}")
    print("="*55 + "\n")
    print(f"📊 Total Data: {len(mentah_data)} baris | Pasaran: {total_jenis_pasaran}")

    # ==============================================
    # LATIH MARKOV
    # ==============================================
    markov = MarkovModel()
    markov.latih([d["angka"] for d in mentah_data])
    print(f"🔗 Markov 1 & 2 → Terlatih")

    # ==============================================
    # BANGUN LSTM
    # ==============================================
    input_angka = Input(shape=(LOOKBACK, 4), name='input_angka')
    lstm_layer = LSTM(64, activation='relu', return_sequences=False)(input_angka)
    input_pasaran = Input(shape=(1,), name='input_pasaran')
    emb_pasaran = Embedding(input_dim=total_jenis_pasaran, output_dim=8)(input_pasaran)
    flat_pasaran = Flatten()(emb_pasaran)
    gabungan_fitur = concatenate([lstm_layer, flat_pasaran])
    dense_shared = Dense(32, activation='relu')(gabungan_fitur)

    out_as  = Dense(10, activation='softmax', name='output_as')(dense_shared)
    out_kop = Dense(10, activation='softmax', name='output_kop')(dense_shared)
    out_kep = Dense(10, activation='softmax', name='output_kep')(dense_shared)
    out_eko = Dense(10, activation='softmax', name='output_eko')(dense_shared)

    model = Model(inputs=[input_angka, input_pasaran], outputs=[out_as, out_kop, out_kep, out_eko])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # ==============================================
    # SIAPKAN DATA LATIH
    # ==============================================
    total_sampel = len(mentah_data) - LOOKBACK
    if total_sampel < 1:
        print("⚠️ Data belum cukup!")
        return

    X_angka = np.zeros((total_sampel, LOOKBACK, 4), dtype=np.float32)
    X_konteks = np.zeros(total_sampel, dtype=np.int32)
    Y = np.zeros((total_sampel, 4), dtype=np.int32)

    for i in range(total_sampel):
        X_angka[i] = [mentah_data[j]["angka"] for j in range(i, i + LOOKBACK)]
        X_konteks[i] = pasaran_ke_idx[mentah_data[i + LOOKBACK]["pasaran"]]
        Y[i] = mentah_data[i + LOOKBACK]["angka"]

    y_train = [Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]]

    # === LATIH ===
    print(f"🧠 Melatih LSTM (75 Epoch / 512 Batch)...")
    model.fit(
        {'input_angka': X_angka, 'input_pasaran': X_konteks},
        y_train,
        epochs=50,
        batch_size=250,
        verbose=0
    )

    # === INPUT TERBARU ===
    input_terbaru = np.array(
        [mentah_data[j]["angka"] for j in range(-LOOKBACK, 0)],
        dtype=np.float32
    )
    input_terbaru = np.expand_dims(input_terbaru, axis=0)

    deret_terakhir = []
    for d in mentah_data[-LOOKBACK:]:
        deret_terakhir.extend(d["angka"])
    satu_terakhir = deret_terakhir[-1]
    dua_terakhir = deret_terakhir[-2:]

    # ==============================================
    # BOBOT GABUNGAN
    # ==============================================
    BOBOT_LSTM   = 0.60
    BOBOT_MARKOV1 = 0.25
    BOBOT_MARKOV2 = 0.15

    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "total_data": len(mentah_data),
        "daftar_pasaran": list_pasaran,
        "data_terbaru": data_terbaru_per_pasaran,
        "metode": f"LSTM {BOBOT_LSTM} + MARKOV1 {BOBOT_MARKOV1} + MARKOV2 {BOBOT_MARKOV2}",
        "seed": SEED_TETAP,
        "hasil": {}
    }

    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]
    print("\n🎯 PREDIKSI PER PASARAN")
    print("="*70)
    for p in list_pasaran:
        idx_target = np.array([pasaran_ke_idx[p]])
        pred_lstm = model.predict({'input_angka': input_terbaru, 'input_pasaran': idx_target}, verbose=0)

        posisi_data = {}
        dtr = data_terbaru_per_pasaran[p]
        print(f"\n📌 {p:8} | Terakhir: {dtr['nomor']} | {dtr['tanggal']}")
        for idx_pos, nama in enumerate(nama_posisi):
            skor_lstm = pred_lstm[idx_pos][0]
            skor_mkv1 = markov.prediksi_orde1(satu_terakhir)
            skor_mkv2 = markov.prediksi_orde2(dua_terakhir)

            skor_gabung = (
                skor_lstm   * BOBOT_LSTM   +
                skor_mkv1   * BOBOT_MARKOV1 +
                skor_mkv2   * BOBOT_MARKOV2
            )

            urut = np.argsort(skor_gabung)[::-1].tolist()
            sembilan = urut[:9]
            satu_hilang = urut[9]
            tujuh = urut[:6] + [satu_hilang]

            posisi_data[nama] = {
                "lima": [str(a) for a in tujuh],
                "sembilan": [str(a) for a in sembilan],
                "skor_tertinggi": f"{skor_gabung[urut[0]]:.4f}"
            }
            print(f"   {nama:6} | 7D: {''.join(posisi_data[nama]['lima'])}  | 9D: {''.join(posisi_data[nama]['sembilan'])}")
        hasil_akhir["hasil"][p] = posisi_data

    # === SIMPAN ===
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print(f"✅ Selesai → hasil_prediksi.json")
    print(f"🔒 Seed: {SEED_TETAP} | Ubah seed jika ada data baru")

if __name__ == "__main__":
    proses_semua()
