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
from tensorflow.keras.layers import LSTM, Dense, Input

# === PENGATURAN ===
DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10           # Panjang urutan riwayat per pasaran
LIMIT_PER_PASARAN = 2000 # Maksimal baris TERBARU yang dipakai per pasaran
FAKTOR_OVERDUE = 0.40

def hitung_bobot_overdue(data_pasaran, posisi_idx):
    """Hitung bobot: makin lama tidak muncul → bobot makin besar"""
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

def format_hasil(prob):
    urut = np.argsort(prob)[::-1].tolist()
    sembilan = urut[:9]
    tujuh = urut[:7] + [urut[9]]
    return {
        "tujuh": [str(a) for a in tujuh],
        "sembilan": [str(a) for a in sembilan]
    }

def bangun_model(ukuran_urutan=LOOKBACK):
    input_seq = Input(shape=(ukuran_urutan, 4), name='urutan_angka')
    x = LSTM(64, activation='relu', return_sequences=False)(input_seq)
    x = Dense(32, activation='relu')(x)
    out_as  = Dense(10, activation='softmax', name='as')(x)
    out_kop = Dense(10, activation='softmax', name='kop')(x)
    out_kep = Dense(10, activation='softmax', name='kep')(x)
    out_eko = Dense(10, activation='softmax', name='eko')(x)
    
    mdl = Model(inputs=input_seq, outputs=[out_as, out_kop, out_kep, out_eko])
    mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return mdl

def proses_semua():
    # === BACA DATA ===
    req = urllib.request.Request(
        DATA_UNDIAN_URL,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        isi_file = r.read().decode('utf-8')
    
    mentah_data = []
    for baris in isi_file.strip().splitlines():
        bagian = baris.split('|')
        if len(bagian) < 4:
            continue
        angka_4d = bagian[2].strip()
        if len(angka_4d) == 4 and angka_4d.isdigit():
            mentah_data.append({
                "pasaran": bagian[0].strip().upper(),
                "tanggal": bagian[1].strip(),
                "angka": [int(d) for d in angka_4d],
                "nomor": angka_4d,
                "waktu": bagian[3].strip()
            })
    
    # ✅ Urutkan naik: lama dulu → baru kemudian
    mentah_data.sort(key=lambda x: (x['tanggal'], x['waktu']))
    
    # ✅ PISAH PER PASARAN + POTONG LIMIT PER PASARAN
    data_per_pasaran = {}
    for baris in mentah_data:
        p = baris['pasaran']
        if p not in data_per_pasaran:
            data_per_pasaran[p] = []
        data_per_pasaran[p].append(baris)
    
    # ✅ Potong masing-masing ambil yang TERBARU saja
    for p in list(data_per_pasaran.keys()):
        if len(data_per_pasaran[p]) > LIMIT_PER_PASARAN:
            data_per_pasaran[p] = data_per_pasaran[p][-LIMIT_PER_PASARAN:]
    
    list_pasaran = sorted(data_per_pasaran.keys())
    data_terbaru_per_pasaran = {p: data_per_pasaran[p][-1] for p in list_pasaran}
    
    # 📋 DATA TERBARU
    print("\n" + "="*60)
    print("📋 DATA TERBARU PER PASARAN")
    print("="*60)
    for p in list_pasaran:
        d = data_terbaru_per_pasaran[p]
        print(f" {p:8} | {d['nomor']:4} | {d['tanggal']} {d['waktu'][-8:]} | total: {len(data_per_pasaran[p])} baris")
    print("="*60 + "\n")
    
    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "limit_per_pasaran": LIMIT_PER_PASARAN,
        "daftar_pasaran": list_pasaran,
        "data_terbaru": {p: {"nomor":d['nomor'],"tanggal":d['tanggal'],"waktu":d['waktu']} for p,d in data_terbaru_per_pasaran.items()},
        "seed": SEED_TETAP,
        "pengaturan": f"LOOKBACK={LOOKBACK} | LIMIT_PER_PASARAN={LIMIT_PER_PASARAN} | OVERDUE={FAKTOR_OVERDUE*100:.0f}%",
        "hasil": {}
    }
    
    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]
    print("═"*70)
    print("🎯 PERBANDINGAN: LSTM MURNI  vs  LSTM + OVERDUE")
    print("═"*70)
    
    for p in list_pasaran:
        dp = data_per_pasaran[p]
        total = len(dp)
        
        # Syarat minimal: LOOKBACK + 1
        if total < LOOKBACK + 1:
            print(f"\n⚠️ {p:8} — dilewati: butuh minimal {LOOKBACK+1} baris, punya {total}")
            continue
        
        print(f"\n📌 {p:8} | Terakhir: {dp[-1]['nomor']} | {dp[-1]['tanggal']} | dipakai: {total} baris")
        
        # === SIAPKAN DATA KHUSUS PASARAN INI ===
        sampel = total - LOOKBACK
        X = np.zeros((sampel, LOOKBACK, 4), dtype=np.float32)
        Y = np.zeros((sampel, 4), dtype=np.int32)
        
        for i in range(sampel):
            X[i] = [dp[j]['angka'] for j in range(i, i + LOOKBACK)]
            Y[i] = dp[i + LOOKBACK]['angka']
        
        y_output = [Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]]
        
        # === LATIH MODEL KHUSUS PASARAN INI ===
        print(f"   🧠 Melatih LSTM... sampel: {sampel}")
        model = bangun_model(LOOKBACK)
        model.fit(X, y_output, epochs=50, batch_size=32, verbose=0)
        
        # === INPUT PREDIKSI = LOOKBACK baris TERAKHIR pasaran ini saja ===
        input_terbaru = np.array(
            [dp[j]['angka'] for j in range(-LOOKBACK, 0)],
            dtype=np.float32
        )
        input_terbaru = np.expand_dims(input_terbaru, axis=0)
        
        pred = model.predict(input_terbaru, verbose=0)
        
        hasil_akhir["hasil"][p] = {}
        for idx_pos, nama in enumerate(nama_posisi):
            prob_lstm = pred[idx_pos][0].copy()
            
            # LSTM MURNI
            hasil_lstm = format_hasil(prob_lstm)
            
            # LSTM + OVERDUE + NORMALISASI
            bobot = hitung_bobot_overdue(dp, idx_pos)
            prob_gabung = prob_lstm.copy()
            for d in range(10):
                prob_gabung[d] *= bobot[str(d)]
            prob_gabung /= np.sum(prob_gabung)
            
            hasil_gabung = format_hasil(prob_gabung)
            
            print(f"   {nama:8}")
            print(f"      LSTM MURNI  | 8D: {''.join(hasil_lstm['tujuh'])}  | 9D: {''.join(hasil_lstm['sembilan'])}")
            print(f"      +OVERDUE    | 8D: {''.join(hasil_gabung['tujuh'])}  | 9D: {''.join(hasil_gabung['sembilan'])}")
            
            hasil_akhir["hasil"][p][nama] = {
                "lstm_murni": hasil_lstm,
                "lstm_overdue": hasil_gabung
            }
    
    # === SIMPAN ===
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)
    
    print("\n" + "═"*70)
    print(f"✅ Selesai → hasil_prediksi.json")
    print(f"🔒 Seed: {SEED_TETAP}")

if __name__ == "__main__":
    proses_semua()
