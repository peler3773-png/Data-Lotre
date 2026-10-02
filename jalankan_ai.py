# jalankan_ai.py — Otomatis Semua Pasaran dari Data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import urllib.request
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"

def proses_semua():
    # === AMBIL DATA ===
    req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        isi_file = r.read().decode('utf-8')

    mentah_data = []
    semua_pasaran = set()
    
    for baris in isi_file.strip().split('\n'):
        if not baris.strip():
            continue
        bagian = baris.split('|')
        if len(bagian) >= 3:
            pasaran = bagian[0].strip().upper()
            angka_4d = bagian[2].strip()
            if len(angka_4d) == 4 and angka_4d.isdigit():
                mentah_data.append({"pasaran": pasaran, "angka": [int(d) for d in angka_4d]})
                semua_pasaran.add(pasaran)

    LIMIT_DATA = 80000
    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[-LIMIT_DATA:]

    # === SEMUA PASARAN OTOMATIS DARI DATA ===
    list_pasaran = sorted(list(semua_pasaran))  # ✅ Otomatis dari data, tidak dibatasi
    pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)
    lookback = 10

    print(f"📊 Ditemukan {total_jenis_pasaran} jenis pasaran dari data:")
    print(f"   {', '.join(list_pasaran)}\n")

    # === BANGUN MODEL LSTM ===
    input_angka = Input(shape=(lookback, 4), name='input_angka')
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

    # === SIAPKAN DATA LATIHAN ===
    X_angka, X_konteks, Y = [], [], []
    for i in range(len(mentah_data) - lookback):
        X_angka.append([d["angka"] for d in mentah_data[i:i+lookback]])
        X_konteks.append(pasaran_ke_idx[mentah_data[i+lookback]["pasaran"]])
        Y.append(mentah_data[i+lookback]["angka"])

    X_angka = np.array(X_angka, dtype=np.float32)
    X_konteks = np.array(X_konteks, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)
    y_train = [Y[:,0], Y[:,1], Y[:,2], Y[:,3]]

    # === LATIH MODEL ===
    print("🧠 Melatih model LSTM...")
    model.fit(
        {'input_angka': X_angka, 'input_pasaran': X_konteks},
        y_train,
        epochs=60,
        batch_size=256,
        validation_split=0.05,
        verbose=1
    )

    # === DATA TERBARU ===
    input_terbaru = np.array([d["angka"] for d in mentah_data[-lookback:]], dtype=np.float32)
    input_terbaru = np.expand_dims(input_terbaru, axis=0)

    # === PREDIKSI SEMUA PASARAN OTOMATIS ===
    hasil_akhir = {
        "diperbarui": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_data": len(mentah_data),
        "daftar_pasaran": list_pasaran,  # ✅ Daftar lengkap untuk Blogger
        "hasil": {}
    }

    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]
    for p in list_pasaran:
        idx_target = np.array([pasaran_ke_idx[p]])
        pred = model.predict({'input_angka': input_terbaru, 'input_pasaran': idx_target}, verbose=0)
        posisi_data = {}
        for idx_pos, nama in enumerate(nama_posisi):
            urut = np.argsort(pred[idx_pos][0])[::-1]
            posisi_data[nama] = {
                "lima": sorted([str(a) for a in urut[:7]]),
                "sembilan": [str(a) for a in urut[:9]]
            }
        hasil_akhir["hasil"][p] = posisi_data

    # === SIMPAN HASIL ===
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print(f"✅ Selesai! Diproses: {len(list_pasaran)} pasaran → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
