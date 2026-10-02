# jalankan_ai.py — Epochs 75 + Batch 512
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10
LIMIT_DATA = 80000

def proses_semua():
    # === AMBIL DATA ===
    req = urllib.request.Request(
        DATA_UNDIAN_URL,
        headers={'User-Agent': 'Mozilla/5.0'},
        timeout=60
    )
    with urllib.request.urlopen(req) as r:
        isi_file = r.read().decode('utf-8')

    mentah_data = []
    semua_pasaran = set()

    for baris in isi_file.strip().splitlines():
        bagian = baris.split('|')
        if len(bagian) < 3:
            continue
        pasaran = bagian[0].strip().upper()
        angka_4d = bagian[2].strip()
        if len(angka_4d) == 4 and angka_4d.isdigit():
            mentah_data.append({
                "pasaran": pasaran,
                "angka": [int(d) for d in angka_4d]
            })
            semua_pasaran.add(pasaran)

    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[-LIMIT_DATA:]

    list_pasaran = sorted(semua_pasaran)
    pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)

    print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] Ditemukan {total_jenis_pasaran} pasaran")

    # === BANGUN MODEL LSTM ===
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

    # === SIAPKAN DATA ===
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

    # === LATIH MODEL — EPOCHS 75, BATCH 512 ===
# jalankan_ai.py — Epochs 75 + Batch 512
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10
LIMIT_DATA = 80000

def proses_semua():
    # === AMBIL DATA ===
    req = urllib.request.Request(
        DATA_UNDIAN_URL,
        headers={'User-Agent': 'Mozilla/5.0'},
        timeout=60
    )
    with urllib.request.urlopen(req) as r:
        isi_file = r.read().decode('utf-8')

    mentah_data = []
    semua_pasaran = set()

    for baris in isi_file.strip().splitlines():
        bagian = baris.split('|')
        if len(bagian) < 3:
            continue
        pasaran = bagian[0].strip().upper()
        angka_4d = bagian[2].strip()
        if len(angka_4d) == 4 and angka_4d.isdigit():
            mentah_data.append({
                "pasaran": pasaran,
                "angka": [int(d) for d in angka_4d]
            })
            semua_pasaran.add(pasaran)

    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[-LIMIT_DATA:]

    list_pasaran = sorted(semua_pasaran)
    pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)

    print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] Ditemukan {total_jenis_pasaran} pasaran")

    # === BANGUN MODEL LSTM ===
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

    # === SIAPKAN DATA ===
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

    # === LATIH MODEL — EPOCHS 75, BATCH 512 ===
    print(f"🧠 [{datetime.now().strftime('%H:%M:%S')}] Melatih model...")
    model.fit(
        {'input_angka': X_angka, 'input_pasaran': X_konteks},
        y_train,
        epochs=75,
        batch_size=512,
        verbose=0
    )

    # === DATA TERBARU ===
    input_terbaru = np.array(
        [mentah_data[j]["angka"] for j in range(-LOOKBACK, 0)],
        dtype=np.float32
    )
    input_terbaru = np.expand_dims(input_terbaru, axis=0)

    # === PREDIKSI DENGAN ATURAN ===
    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_data": len(mentah_data),
        "daftar_pasaran": list_pasaran,
        "hasil": {}
    }

    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]

    for p in list_pasaran:
        idx_target = np.array([pasaran_ke_idx[p]])
        pred = model.predict({'input_angka': input_terbaru, 'input_pasaran': idx_target}, verbose=0)
        posisi_data = {}

        for idx_pos, nama in enumerate(nama_posisi):
            urut = np.argsort(pred[idx_pos][0])[::-1].tolist()
            sembilan = urut[:9]
            satu_hilang = urut[9]
            tujuh = urut[:6] + [satu_hilang]

            posisi_data[nama] = {
                "lima": [str(a) for a in tujuh],
                "sembilan": [str(a) for a in sembilan]
            }

        hasil_akhir["hasil"][p] = posisi_data

    # === SIMPAN ===
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()

    # === DATA TERBARU ===
    input_terbaru = np.array(
        [mentah_data[j]["angka"] for j in range(-LOOKBACK, 0)],
        dtype=np.float32
    )
    input_terbaru = np.expand_dims(input_terbaru, axis=0)

    # === PREDIKSI DENGAN ATURAN ===
    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_data": len(mentah_data),
        "daftar_pasaran": list_pasaran,
        "hasil": {}
    }

    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]

    for p in list_pasaran:
        idx_target = np.array([pasaran_ke_idx[p]])
        pred = model.predict({'input_angka': input_terbaru, 'input_pasaran': idx_target}, verbose=0)
        posisi_data = {}

        for idx_pos, nama in enumerate(nama_posisi):
            urut = np.argsort(pred[idx_pos][0])[::-1].tolist()
            sembilan = urut[:9]
            satu_hilang = urut[9]
            tujuh = urut[:6] + [satu_hilang]

            posisi_data[nama] = {
                "lima": [str(a) for a in tujuh],
                "sembilan": [str(a) for a in sembilan]
            }

        hasil_akhir["hasil"][p] = posisi_data

    # === SIMPAN ===
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
