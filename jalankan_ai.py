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
LIMIT_DATA = 80000
LOOKBACK = 2

def proses_semua():
    # === AMBIL DATA ===
    req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        isi_file = r.read().decode('utf-8')
    
    semua_data = []
    semua_pasaran = set()
    
    for baris in isi_file.strip().split('\n'):
        if not baris.strip():
            continue
        bagian = baris.split('|')
        if len(bagian) >= 3:
            pasaran = bagian[0].strip().upper()
            angka_4d = bagian[2].strip()
            if len(angka_4d) == 4 and angka_4d.isdigit():
                semua_data.append({"pasaran": pasaran, "angka": [int(d) for d in angka_4d]})
                semua_pasaran.add(pasaran)
    
    if len(semua_data) > LIMIT_DATA:
        semua_data = semua_data[-LIMIT_DATA:]
    
    list_pasaran = sorted(list(semua_pasaran))
    pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)
    
    print(f"📊 Ditemukan {total_jenis_pasaran} pasaran, total {len(semua_data)} baris data\n")
    
    # === FUNGSI PROSES DIGIT HILANG ===
    def buat_hasil(urut):
        semua_d = set('0123456789')
        d9 = [str(a) for a in urut[:9]]
        hilang = sorted(semua_d - set(d9))
        d7 = [str(a) for a in urut[:7]]
        return ''.join(hilang + d7), ''.join(d9)
    
    # === JIKA DATA KURANG ===
    if len(semua_data) < LOOKBACK + 1:
        print("⚠️ Data terbatas → pakai pola dasar\n")
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hasil_json = {"diperbarui": waktu, "hasil": {}}
        for p in list_pasaran:
            g = {"AS":"01234567","KOP":"12345678","KEPALA":"23456789","EKOR":"34567890"}
            d9 = {"AS":"123456789","KOP":"234567890","KEPALA":"345678901","EKOR":"456789012"}
            hasil_json["hasil"][p] = {"gabung":g,"9digit":d9}
            print(f"{'='*40}")
            print(f"            POLA TARUNG")
            print(f"{'='*40}")
            print(f"PASARAN   : {p}")
            print(f"UPDATE    : {waktu}")
            print(f"DATA      : {LIMIT_DATA} BARIS DIPELAJARI")
            print(f"{'='*40}")
            for pos in ["AS","KOP","KEPALA","EKOR"]:
                print(f"POSISI {pos}:")
                print(f"  7 DIGIT : {g[pos]}")
                print(f"  9 DIGIT : {d9[pos]}")
            print(f"  7 DIGIT : {g['AS']},{g['KOP']},{g['KEPALA']},{g['EKOR']}")
            print(f"  9 DIGIT : {d9['AS']},{d9['KOP']},{d9['KEPALA']},{d9['EKOR']}")
            print(f"{'='*40}\n")
        with open("hasil_prediksi.json","w",encoding="utf-8") as f:
            json.dump(hasil_json,f,ensure_ascii=False,indent=2)
        print("✅ Selesai → hasil_prediksi.json")
        return
    
    # === SIAPKAN DATA LATIHAN ===
    X_a, X_k, Y = [], [], []
    for i in range(len(semua_data)-LOOKBACK):
        X_a.append([d["angka"] for d in semua_data[i:i+LOOKBACK]])
        X_k.append(pasaran_ke_idx[semua_data[i+LOOKBACK]["pasaran"]])
        Y.append(semua_data[i+LOOKBACK]["angka"])
    
    X_a = np.array(X_a, dtype=np.float32)
    X_k = np.array(X_k, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)
    y_train = [Y[:,0], Y[:,1], Y[:,2], Y[:,3]]
    
    # === BANGUN MODEL ===
    in_angka = Input(shape=(LOOKBACK,4))
    in_psn = Input(shape=(1,))
    x = LSTM(16, activation='relu')(in_angka)
    e = Flatten()(Embedding(total_jenis_pasaran,4)(in_psn))
    x = concatenate([x,e])
    x = Dense(16, activation='relu')(x)
    o1 = Dense(10, activation='softmax')(x)
    o2 = Dense(10, activation='softmax')(x)
    o3 = Dense(10, activation='softmax')(x)
    o4 = Dense(10, activation='softmax')(x)
    
    model = Model([in_angka,in_psn],[o1,o2,o3,o4])
    model.compile('adam','sparse_categorical_crossentropy')
    
    print("🧠 Melatih...\n")
    model.fit([X_a,X_k], y_train, epochs=20, batch_size=32, verbose=0)
    
    # === PREDIKSI ===
    input_terbaru = np.array([d["angka"] for d in semua_data[-LOOKBACK:]], dtype=np.float32)
    input_terbaru = np.expand_dims(input_terbaru, axis=0)
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hasil_json = {"diperbarui": waktu, "hasil": {}}
    
    for p in list_pasaran:
        pred = model.predict([input_terbaru, np.array([pasaran_ke_idx[p]])], verbose=0)
        g, d9 = {}, {}
        for i,pos in enumerate(["AS","KOP","KEPALA","EKOR"]):
            urut = np.argsort(pred[i][0])[::-1]
            g[pos], d9[pos] = buat_hasil(urut)
        hasil_json["hasil"][p] = {"gabung":g,"9digit":d9}
        
        print(f"{'='*40}")
        print(f"            POLA TARUNG")
        print(f"{'='*40}")
        print(f"PASARAN   : {p}")
        print(f"UPDATE    : {waktu}")
        print(f"DATA      : {LIMIT_DATA} BARIS DIPELAJARI")
        print(f"{'='*40}")
        for pos in ["AS","KOP","KEPALA","EKOR"]:
            print(f"POSISI {pos}:")
            print(f"  7 DIGIT : {g[pos]}")
            print(f"  9 DIGIT : {d9[pos]}")
        print(f"  7 DIGIT : {g['AS']},{g['KOP']},{g['KEPALA']},{g['EKOR']}")
        print(f"  9 DIGIT : {d9['AS']},{d9['KOP']},{d9['KEPALA']},{d9['EKOR']}")
        print(f"{'='*40}\n")
    
    with open("hasil_prediksi.json","w",encoding="utf-8") as f:
        json.dump(hasil_json,f,ensure_ascii=False,indent=2)
    print("✅ Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
