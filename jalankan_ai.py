import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 🔒 KUNCI ACAK
SEED_TETAP = 20261003
import random
random.seed(SEED_TETAP)
import numpy as np
np.random.seed(SEED_TETAP)
import tensorflow as tf
tf.random.set_seed(SEED_TETAP)
tf.get_logger().setLevel('ERROR')

# Cek & pasang Optuna
try:
    import optuna
except ImportError:
    print("📦 Menginstal Optuna...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
    import optuna

import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === PENGATURAN ===
DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 10
LIMIT_PER_PASARAN = 2000
FAKTOR_OVERDUE = 0.40

# Nilai Manual
MANUAL_EPOCH = 50
MANUAL_BATCH = 32

# Rentang pencarian Optuna
OPTUNA_EPOCH_MIN = 20
OPTUNA_EPOCH_MAX = 80
OPTUNA_BATCH_CHOICES = [16, 32, 64]
OPTUNA_CUPIKAN = 12  # Jumlah percobaan per pasaran

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

def format_hasil(prob):
    urut = np.argsort(prob)[::-1].tolist()
    sembilan = urut[:9]
    tujuh = urut[:7] + [urut[9]]
    return {
        "tujuh": [str(a) for a in tujuh],
        "sembilan": [str(a) for a in sembilan]
    }

def bangun_model(ukuran_urutan=LOOKBACK):
    input_seq = Input(shape=(ukuran_urutan, 4))
    x = LSTM(64, activation='relu', return_sequences=False)(input_seq)
    x = Dense(32, activation='relu')(x)
    out_as  = Dense(10, activation='softmax', name='as')(x)
    out_kop = Dense(10, activation='softmax', name='kop')(x)
    out_kep = Dense(10, activation='softmax', name='kep')(x)
    out_eko = Dense(10, activation='softmax', name='eko')(x)
    mdl = Model(inputs=input_seq, outputs=[out_as, out_kop, out_kep, out_eko])
    mdl.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return mdl

def siapkan_data(dp):
    total = len(dp)
    sampel = total - LOOKBACK
    X = np.zeros((sampel, LOOKBACK, 4), dtype=np.float32)
    Y = np.zeros((sampel, 4), dtype=np.int32)
    for i in range(sampel):
        X[i] = [dp[j]['angka'] for j in range(i, i + LOOKBACK)]
        Y[i] = dp[i + LOOKBACK]['angka']
    y_output = [Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]]
    # Bagi latih 90% + validasi 10%
    batas = int(0.9 * sampel)
    return X[:batas], y_output[:], X[batas:], y_output[:], batas

def latih_manual(X, Y, Xv, Yv):
    model = bangun_model()
    model.fit(X, Y, epochs=MANUAL_EPOCH, batch_size=MANUAL_BATCH,
              validation_data=(Xv, Yv), verbose=0)
    return model, {"epoch": MANUAL_EPOCH, "batch_size": MANUAL_BATCH}

def latih_earlystop(X, Y, Xv, Yv):
    model = bangun_model()
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    hist = model.fit(X, Y, epochs=100, batch_size=MANUAL_BATCH,
                     validation_data=(Xv, Yv), callbacks=[es], verbose=0)
    dipakai = len(hist.history['loss']) - es.patience
    return model, {"epoch": max(5, dipakai), "batch_size": MANUAL_BATCH, "berhenti_di": len(hist.history['loss'])}

def cari_optuna(X, Y, Xv, Yv):
    def tujuan(trial):
        epoch = trial.suggest_int('epoch', OPTUNA_EPOCH_MIN, OPTUNA_EPOCH_MAX)
        batch = trial.suggest_categorical('batch_size', OPTUNA_BATCH_CHOICES)
        m = bangun_model()
        es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        h = m.fit(X, Y, epochs=epoch, batch_size=batch,
                  validation_data=(Xv, Yv), callbacks=[es], verbose=0)
        return min(h.history['val_loss'])
    
    study = optuna.create_study(direction='minimize')
    study.optimize(tujuan, n_trials=OPTUNA_CUPIKAN, show_progress_bar=False)
    bp = study.best_params
    # Latih ulang dengan nilai terbaik
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    model = bangun_model()
    model.fit(X, Y, epochs=bp['epoch'], batch_size=bp['batch_size'],
              validation_data=(Xv, Yv), callbacks=[es], verbose=0)
    return model, {"epoch": bp['epoch'], "batch_size": bp['batch_size'], "skor_terbaik": study.best_value}

def proses_semua():
    # Baca data
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
    
    mentah_data.sort(key=lambda x: (x['tanggal'], x['waktu']))
    
    # Pisah per pasaran
    data_per_pasaran = {}
    for baris in mentah_data:
        p = baris['pasaran']
        if p not in data_per_pasaran:
            data_per_pasaran[p] = []
        data_per_pasaran[p].append(baris)
    
    # Potong batas
    for p in list(data_per_pasaran.keys()):
        if len(data_per_pasaran[p]) > LIMIT_PER_PASARAN:
            data_per_pasaran[p] = data_per_pasaran[p][-LIMIT_PER_PASARAN:]
    
    list_pasaran = sorted(data_per_pasaran.keys())
    data_terbaru = {p: data_per_pasaran[p][-1] for p in list_pasaran}
    
    # Tampilkan daftar
    print("\n" + "="*70)
    print("📋 DATA & 3 METODE: MANUAL | EARLY STOPPING | OPTUNA")
    print("="*70)
    for p in list_pasaran:
        d = data_terbaru[p]
        print(f" {p:8} | Terakhir: {d['nomor']} | Total baris: {len(data_per_pasaran[p])}")
    print(f"\n Pengaturan: LOOKBACK={LOOKBACK} | LIMIT={LIMIT_PER_PASARAN}")
    print(f" Manual: EPOCH={MANUAL_EPOCH} BATCH={MANUAL_BATCH}")
    print(f" Optuna: EPOCH {OPTUNA_EPOCH_MIN}-{OPTUNA_EPOCH_MAX} BATCH {OPTUNA_BATCH_CHOICES} | {OPTUNA_CUPIKAN} percobaan")
    print("="*70)
    
    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "pengaturan": {
            "LOOKBACK": LOOKBACK,
            "LIMIT_PER_PASARAN": LIMIT_PER_PASARAN,
            "manual": {"epoch": MANUAL_EPOCH, "batch_size": MANUAL_BATCH},
            "optuna": {"epoch_min": OPTUNA_EPOCH_MIN, "epoch_max": OPTUNA_EPOCH_MAX,
                       "batch_choices": OPTUNA_BATCH_CHOICES, "percobaan": OPTUNA_CUPIKAN}
        },
        "daftar_pasaran": list_pasaran,
        "hasil": {}
    }
    
    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]
    
    for p in list_pasaran:
        dp = data_per_pasaran[p]
        total = len(dp)
        
        if total < LOOKBACK + 1:
            print(f"\n⚠️ {p:8} — dilewati: butuh minimal {LOOKBACK+1} baris, punya {total}")
            continue
        
        print(f"\n{'─'*70}")
        print(f"🔄 MEMPROSES: {p} | {total} baris")
        print(f"{'─'*70}")
        
        # Siapkan data
        X, Y, Xv, Yv, batas = siapkan_data(dp)
        input_terbaru = np.array([dp[j]['angka'] for j in range(-LOOKBACK, 0)], dtype=np.float32)
        input_terbaru = np.expand_dims(input_terbaru, axis=0)
        
        hasil_pasaran = {}
        
        # ─── 1. MANUAL ───
        print(f"  ▶️  1/3 MANUAL (EPOCH={MANUAL_EPOCH}, BATCH={MANUAL_BATCH})...")
        model, info = latih_manual(X, Y, Xv, Yv)
        pred = model.predict(input_terbaru, verbose=0)
        hasil_m = {}
        hasil_m["pengaturan"] = info
        for idx, nm in enumerate(nama_posisi):
            prob = pred[idx][0].copy()
            prob /= np.sum(prob)
            bobot = hitung_bobot_overdue(dp, idx)
            prob_ov = prob.copy()
            for d in range(10):
                prob_ov[d] *= bobot[str(d)]
            prob_ov /= np.sum(prob_ov)
            hasil_m[nm] = {
                "murni": format_hasil(prob),
                "overdue": format_hasil(prob_ov)
            }
        hasil_pasaran["manual"] = hasil_m
        print(f"     ✅ Selesai")
        
        # ─── 2. EARLY STOPPING ───
        print(f"  ▶️  2/3 EARLY STOPPING...")
        model, info = latih_earlystop(X, Y, Xv, Yv)
        pred = model.predict(input_terbaru, verbose=0)
        hasil_e = {}
        hasil_e["pengaturan"] = info
        for idx, nm in enumerate(nama_posisi):
            prob = pred[idx][0].copy()
            prob /= np.sum(prob)
            bobot = hitung_bobot_overdue(dp, idx)
            prob_ov = prob.copy()
            for d in range(10):
                prob_ov[d] *= bobot[str(d)]
            prob_ov /= np.sum(prob_ov)
            hasil_e[nm] = {
                "murni": format_hasil(prob),
                "overdue": format_hasil(prob_ov)
            }
        hasil_pasaran["early_stopping"] = hasil_e
        print(f"     ✅ Berhenti di epoch {info['berhenti_di']}")
        
        # ─── 3. OPTUNA ───
        print(f"  ▶️  3/3 OPTUNA ({OPTUNA_CUPIKAN} percobaan)...")
        model, info = cari_optuna(X, Y, Xv, Yv)
        pred = model.predict(input_terbaru, verbose=0)
        hasil_o = {}
        hasil_o["pengaturan"] = info
        for idx, nm in enumerate(nama_posisi):
            prob = pred[idx][0].copy()
            prob /= np.sum(prob)
            bobot = hitung_bobot_overdue(dp, idx)
            prob_ov = prob.copy()
            for d in range(10):
                prob_ov[d] *= bobot[str(d)]
            prob_ov /= np.sum(prob_ov)
            hasil_o[nm] = {
                "murni": format_hasil(prob),
                "overdue": format_hasil(prob_ov)
            }
        hasil_pasaran["optuna"] = hasil_o
        print(f"     ✅ Terbaik: EPOCH={info['epoch']} BATCH={info['batch_size']}")
        
        # Tampilkan ringkasan
        hasil_akhir["hasil"][p] = hasil_pasaran
        
        # Tampilkan perbandingan untuk EKOR sebagai contoh
        print(f"\n 📊 PERBANDINGAN CONTOH (EKOR +OVERDUE):")
        print(f"    MANUAL    : 8D: {''.join(hasil_m['EKOR']['overdue']['tujuh'])}  9D: {''.join(hasil_m['EKOR']['overdue']['sembilan'])}")
        print(f"    EARLY STOP: 8D: {''.join(hasil_e['EKOR']['overdue']['tujuh'])}  9D: {''.join(hasil_e['EKOR']['overdue']['sembilan'])}")
        print(f"    OPTUNA    : 8D: {''.join(hasil_o['EKOR']['overdue']['tujuh'])}  9D: {''.join(hasil_o['EKOR']['overdue']['sembilan'])}")
    
    # Simpan
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ SEMUA SELESAI → hasil_prediksi.json")
    print(f"   Isi: manual + early_stopping + optuna untuk tiap pasaran")
    print(f"{'='*70}")

if __name__ == "__main__":
    proses_semua()
