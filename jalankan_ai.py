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

def gabungkan_hasil(daftar_7, daftar_9):
    semua_digit = set('0123456789')
    ada_di_9 = set(daftar_9)
    hilang = sorted(semua_digit - ada_di_9)
    return hilang + daftar_7  # hilang duluan, lalu 7 digit urutan asli

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
    
    list_pasaran = sorted(list(semua_pasaran))
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
    model.fit({'input_angka': X_angka, 'input_pasaran': X_konteks},
              y_train, epochs=45, batch_size=128, verbose=0)
    
    # === DATA TERBARU ===
    input_terbaru = np.array([d["angka"] for d in mentah_data[-lookback:]], dtype=np.float32)
    input_terbaru = np.expand_dims(input_terbaru, axis=0)
    
    # === PREDIKSI SEMUA PASARAN ===
    hasil_akhir = {
        "diperbarui": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            urut = np.argsort(pred[idx_pos][0])[::-1]
            daftar_7 = [str(a) for a in urut[:7]]
            daftar_9 = [str(a) for a in urut[:9]]
            gabung = gabungkan_hasil(daftar_7, daftar_9)
            
            posisi_data[nama] = {
                "gabung": "".join(gabung)  # Tanpa pemisah, langsung rapat
            }
        
        hasil_akhir["hasil"][p] = posisi_data
    
    # === SIMPAN HASIL ===
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Selesai! Diproses: {len(list_pasaran)} pasaran → hasil_prediksi.json")
    print("\n📋 CONTOH HASIL AKHIR:")
    for p in list(list_pasaran[:3]):
        print(f"\n{p}:")
        for pos, val in hasil_akhir["hasil"][p].items():
            print(f"  {pos}: {val['gabung']}")

if __name__ == "__main__":
    proses_semua()
