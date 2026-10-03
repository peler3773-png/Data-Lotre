import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 🔒 KUNCI ACAK — TETAP SAMA
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
FAKTOR_OVERDUE = 0.40   # Jarak digit tunggal
FAKTOR_ORDE1 = 0.40    # Sama dengan Overdue
FAKTOR_ORDE2 = 0.25    # Pasangan berurutan

# ⚡ OVERDUE — digit tunggal, jarak sejak terakhir muncul
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

# ⚡ ORDE 1 — sama dengan Overdue, fungsi terpisah agar mudah ubah
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

# ⚡ ORDE 2 — pasangan digit berurutan
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
    tujuh = urut[:7]
    delapan = urut[:8]
    sembilan = urut[:9]
    return {
        "tujuh": ''.join(str(a) for a in tujuh),
        "delapan": ''.join(str(a) for a in delapan),
        "sembilan": ''.join(str(a) for a in sembilan)
    }

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
            data_terbaru_per_pasaran[pasaran] = {
                "nomor": angka_4d,
                "tanggal": tgl,
                "waktu": waktu
            }

    # Urutkan: lama → baru
    mentah_data = mentah_data[::-1]
    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[-LIMIT_DATA:]

    list_pasaran = sorted(semua_pasaran)
    pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)
    data_per_pasaran = {p: [b for b in mentah_data if b['pasaran'] == p] for p in list_pasaran}

    # 📋 DATA TERBARU
    print("\n" + "="*70)
    print("📋 DATA TERBARU PER PASARAN")
    print("="*70)
    for p in list_pasaran:
        d = data_terbaru_per_pasaran[p]
        print(f" {p:8} | {d['nomor']:4} | {d['tanggal']} {d['waktu'][-8:]}")
    print("="*70)
    print(f"📊 Total Data: {len(mentah_data)} baris | Pasaran: {total_jenis_pasaran}\n")

    # === MODEL LSTM ===
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

    # === DATA LATIH ===
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
    print(f"🧠 Melatih LSTM (50 Epoch / 250 Batch)...")
    model.fit(
        {'input_angka': X_angka, 'input_pasaran': X_konteks},
        y_train,
        epochs=50,
        batch_size=250,
        verbose=0
    )

    # === INPUT TERAKHIR ===
    input_terbaru = np.array(
        [mentah_data[j]["angka"] for j in range(-LOOKBACK, 0)],
        dtype=np.float32
    )
    input_terbaru = np.expand_dims(input_terbaru, axis=0)

    # === HASIL AKHIR ===
    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "total_data": len(mentah_data),
        "daftar_pasaran": list_pasaran,
        "data_terbaru": data_terbaru_per_pasaran,
        "seed": SEED_TETAP,
        "pengaturan": {
            "FAKTOR_OVERDUE": FAKTOR_OVERDUE,
            "FAKTOR_ORDE1": FAKTOR_ORDE1,
            "FAKTOR_ORDE2": FAKTOR_ORDE2
        },
        "hasil": {}
    }

    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]
    print("\n" + "═"*110)
    print("🎯 MONITOR: LSTM  |  OVERDUE  |  ORDE 1  |  ORDE 2")
    print("═"*110)
    print(f"{'METODE':<14} {'7D':<10} {'8D':<10} {'9D':<12}")
    print("─"*110)

    for p in list_pasaran:
        idx_target = np.array([pasaran_ke_idx[p]])
        pred = model.predict({'input_angka': input_terbaru, 'input_pasaran': idx_target}, verbose=0)
        dtr = data_terbaru_per_pasaran[p]
        data_p = data_per_pasaran[p]

        print(f"\n📌 {p:8} | Terakhir: {dtr['nomor']} | {dtr['tanggal']}")
        hasil_akhir["hasil"][p] = {}

        for idx_pos, nama in enumerate(nama_posisi):
            prob_lstm = pred[idx_pos][0].copy()
            res_lstm = format_hasil(prob_lstm)

            # OVERDUE
            prob_ovd = prob_lstm.copy()
            if len(data_p) > 3:
                bobot_ovd = hitung_bobot_overdue(data_p, idx_pos)
                for d in range(10):
                    prob_ovd[d] *= bobot_ovd[str(d)]
                prob_ovd /= prob_ovd.sum()
            res_ovd = format_hasil(prob_ovd)

            # ORDE 1
            prob_o1 = prob_lstm.copy()
            if len(data_p) > 3:
                bobot_o1 = hitung_bobot_orde1(data_p, idx_pos)
                for d in range(10):
                    prob_o1[d] *= bobot_o1[str(d)]
                prob_o1 /= prob_o1.sum()
            res_o1 = format_hasil(prob_o1)

            # ORDE 2
            prob_o2 = prob_lstm.copy()
            if len(data_p) > 5:
                bobot_o2 = hitung_bobot_orde2(data_p, idx_pos)
                for d in range(10):
                    prob_o2[d] *= bobot_o2[str(d)]
                prob_o2 /= prob_o2.sum()
            res_o2 = format_hasil(prob_o2)

            # Tampil rapi
            print(f" 【{nama}】")
            print(f"  LSTM        | {res_lstm['tujuh']:<10} {res_lstm['delapan']:<10} {res_lstm['sembilan']}")
            print(f"  OVERDUE     | {res_ovd['tujuh']:<10} {res_ovd['delapan']:<10} {res_ovd['sembilan']}")
            print(f"  ORDE 1      | {res_o1['tujuh']:<10} {res_o1['delapan']:<10} {res_o1['sembilan']}")
            print(f"  ORDE 2      | {res_o2['tujuh']:<10} {res_o2['delapan']:<10} {res_o2['sembilan']}")
            print("  " + "─"*100)

            hasil_akhir["hasil"][p][nama] = {
                "lstm": res_lstm,
                "overdue": res_ovd,
                "orde1": res_o1,
                "orde2": res_o2
            }

    # === SIMPAN ===
    nama_file = "monitor_lstm_ovd_orde1_orde2.json"
    with open(nama_file, "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print("\n" + "═"*110)
    print(f"✅ SELESAI → Disimpan: {nama_file}")
    print(f"🔒 Seed: {SEED_TETAP}")
    print(f"⚙️ Faktor → OVERDUE:{FAKTOR_OVERDUE:.0%} | ORDE1:{FAKTOR_ORDE1:.0%} | ORDE2:{FAKTOR_ORDE2:.0%}")
    print("═"*110)

if __name__ == "__main__":
    proses_semua()
