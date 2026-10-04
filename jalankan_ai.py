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
import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10
LIMIT_DATA = 16250
FAKTOR_OVERDUE = 0.40
FAKTOR_ORDE1 = 0.40
FAKTOR_ORDE2 = 0.25

# === OVERDUE ===
def hitung_bobot_overdue(data_pasaran, posisi_idx):
    terakhir_muncul = {str(d): 999 for d in range(10)}
    for urutan, baris in enumerate(reversed(data_pasaran)):
        angka = str(baris['angka'][posisi_idx])
        if terakhir_muncul[angka] == 999:
            terakhir_muncul[angka] = urutan
    max_jarak = max(terakhir_muncul.values()) + 1
    bobot = {}
    for d in range(10):
        j = terakhir_muncul[str(d)]
        bobot[str(d)] = 1.0 + (j / max_jarak) * FAKTOR_OVERDUE
    return bobot

# === ORDE 1 ===
def hitung_bobot_orde1(data_pasaran, posisi_idx):
    terakhir_muncul = {str(d): 999 for d in range(10)}
    for urutan, baris in enumerate(reversed(data_pasaran)):
        angka = str(baris['angka'][posisi_idx])
        if terakhir_muncul[angka] == 999:
            terakhir_muncul[angka] = urutan
    max_jarak = max(terakhir_muncul.values()) + 1
    bobot = {}
    for d in range(10):
        j = terakhir_muncul[str(d)]
        bobot[str(d)] = 1.0 + (j / max_jarak) * FAKTOR_ORDE1
    return bobot

# === ORDE 2 ===
def hitung_bobot_orde2(data_pasaran, posisi_idx):
    terakhir_muncul = {str(d): 999 for d in range(10)}
    if len(data_pasaran) < 2:
        return {str(d): 1.0 for d in range(10)}
    pasangan_terakhir = {}
    for urutan, baris in enumerate(reversed(data_pasaran)):
        if urutan >= len(data_pasaran) - 1:
            continue
        digit_sekarang = str(baris['angka'][posisi_idx])
        baris_sebelum = data_pasaran[-(urutan + 2)]
        digit_sebelum = str(baris_sebelum['angka'][posisi_idx])
        kunci = digit_sebelum + digit_sekarang
        if kunci not in pasangan_terakhir:
            pasangan_terakhir[kunci] = urutan
    for urutan, baris in enumerate(reversed(data_pasaran)):
        if urutan >= len(data_pasaran) - 1:
            continue
        digit_sekarang = str(baris['angka'][posisi_idx])
        if terakhir_muncul[digit_sekarang] == 999:
            baris_sebelum = data_pasaran[-(urutan + 2)]
            digit_sebelum = str(baris_sebelum['angka'][posisi_idx])
            kunci = digit_sebelum + digit_sekarang
            jarak = pasangan_terakhir.get(kunci, 999)
            if jarak < terakhir_muncul[digit_sekarang]:
                terakhir_muncul[digit_sekarang] = jarak
    max_jarak = max(terakhir_muncul.values()) + 1
    bobot = {}
    for d in range(10):
        j = terakhir_muncul[str(d)]
        bobot[str(d)] = 1.0 + (j / max_jarak) * FAKTOR_ORDE2
    return bobot

def format_hasil(prob):
    urut = np.argsort(prob)[::-1].tolist()
    return {
        "tujuh": ''.join(str(a) for a in urut[:7]),
        "delapan": ''.join(str(a) for a in urut[:8]),
        "sembilan": ''.join(str(a) for a in urut[:9])
    }

def proses_semua():
    # AMBIL DATA
    req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent': 'Mozilla/5.0'})
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
        tgl, angka_4d, waktu = bagian[1].strip(), bagian[2].strip(), bagian[3].strip()
        if len(angka_4d) == 4 and angka_4d.isdigit():
            arr = [int(d) for d in angka_4d]
            mentah_data.append({"pasaran": pasaran, "tanggal": tgl, "angka": arr, "nomor": angka_4d, "waktu": waktu})
            semua_pasaran.add(pasaran)
            data_terbaru_per_pasaran[pasaran] = {"nomor": angka_4d, "tanggal": tgl, "waktu": waktu}

    mentah_data = mentah_data[::-1]
    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[-LIMIT_DATA:]

    list_pasaran = sorted(semua_pasaran)
    pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)
    data_per_pasaran = {p: [b for b in mentah_data if b['pasaran'] == p] for p in list_pasaran}

    # MODEL LSTM
    input_angka = Input(shape=(LOOKBACK, 4))
    lstm_layer = LSTM(64, activation='relu')(input_angka)
    input_pasaran = Input(shape=(1,))
    emb_pasaran = Flatten()(Embedding(input_dim=total_jenis_pasaran, output_dim=8)(input_pasaran))
    gabungan = concatenate([lstm_layer, emb_pasaran])
    dense = Dense(32, activation='relu')(gabungan)
    out_as = Dense(10, activation='softmax')(dense)
    out_kop = Dense(10, activation='softmax')(dense)
    out_kep = Dense(10, activation='softmax')(dense)
    out_eko = Dense(10, activation='softmax')(dense)

    model = Model(inputs=[input_angka, input_pasaran], outputs=[out_as, out_kop, out_kep, out_eko])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # DATA LATIH
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

    print("🧠 Melatih model...")
    model.fit({'input_1': X_angka, 'input_2': X_konteks}, [Y[:,0], Y[:,1], Y[:,2], Y[:,3]],
              epochs=50, batch_size=250, verbose=0)

    # INPUT TERKINI
    input_terbaru = np.expand_dims(np.array([mentah_data[j]["angka"] for j in range(-LOOKBACK, 0)], dtype=np.float32), axis=0)

    # HASIL
    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "total_data": len(mentah_data),
        "daftar_pasaran": list_pasaran,
        "data_terbaru": data_terbaru_per_pasaran,
        "hasil": {}
    }

    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]
    for p in list_pasaran:
        idx_target = np.array([pasaran_ke_idx[p]])
        pred = model.predict([input_terbaru, idx_target], verbose=0)
        data_p = data_per_pasaran[p]
        hasil_akhir["hasil"][p] = {}

        for idx_pos, nama in enumerate(nama_posisi):
            prob_lstm = pred[idx_pos][0].copy()
            res_lstm = format_hasil(prob_lstm)

            # OVERDUE
            prob_ovd = prob_lstm.copy()
            if len(data_p) > 3:
                bobot_ovd = hitung_bobot_overdue(data_p, idx_pos)
                for d in range(10): prob_ovd[d] *= bobot_ovd[str(d)]
                prob_ovd /= prob_ovd.sum()
            res_ovd = format_hasil(prob_ovd)

            # ORDE 1 — DIJAMIN ADA
            prob_o1 = prob_lstm.copy()
            if len(data_p) > 3:
                bobot_o1 = hitung_bobot_orde1(data_p, idx_pos)
                for d in range(10): prob_o1[d] *= bobot_o1[str(d)]
                prob_o1 /= prob_o1.sum()
            res_o1 = format_hasil(prob_o1)

            # ORDE 2 — DIJAMIN ADA
            prob_o2 = prob_lstm.copy()
            if len(data_p) > 5:
                bobot_o2 = hitung_bobot_orde2(data_p, idx_pos)
                for d in range(10): prob_o2[d] *= bobot_o2[str(d)]
                prob_o2 /= prob_o2.sum()
            res_o2 = format_hasil(prob_o2)

            # SIMPAN SEMUA — TIDAK AKAN KOSONG
            hasil_akhir["hasil"][p][nama] = {
                "lstm": res_lstm,
                "overdue": res_ovd,
                "orde1": res_o1,
                "orde2": res_o2
            }

    # SIMPAN FILE
    nama_file = "hasil_prediksi.json"
    with open(nama_file, "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print(f"\n✅ BERHASIL disimpan ke: {nama_file}")
    print(f"   Cek isi: buka file → hasil → SGP → AS → ada 'orde1' dan 'orde2'")

if __name__ == "__main__":
    proses_semua()
