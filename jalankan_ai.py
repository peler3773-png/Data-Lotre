import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 🔒 KUNCI ACAK — Hasil dapat diulang
SEED_TETAP = 20261004
import random
random.seed(SEED_TETAP)
import numpy as np
np.random.seed(SEED_TETAP)
import tensorflow as tf
tf.random.set_seed(SEED_TETAP)
tf.get_logger().setLevel('ERROR')

try:
    import optuna
except ImportError:
    print("❌ ERROR: Pustaka 'optuna' belum terpasang!")
    print("Pasang: pip install optuna")
    sys.exit(1)

import urllib.request
import json
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# === PENGATURAN ===
DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LOOKBACK = 12
LIMIT_PER_PASARAN = 2000
FAKTOR_OVERDUE = 0.40
OPTUNA_EPOCH_MIN = 25
OPTUNA_EPOCH_MAX = 60
OPTUNA_BATCH_CHOICES = [16, 32, 64]
OPTUNA_CUPIKAN = 5
VALIDASI_MIN = 5

def hitung_bobot_overdue(data_pasaran, posisi_idx):
    terakhir_muncul = {str(d): None for d in range(10)}
    for urutan, baris in enumerate(reversed(data_pasaran)):
        angka = str(baris['angka'][posisi_idx])
        if terakhir_muncul[angka] is None:
            terakhir_muncul[angka] = urutan
    jarak_tercatat = [v for v in terakhir_muncul.values() if v is not None]
    if not jarak_tercatat:
        max_jarak = 1
        for d in terakhir_muncul:
            terakhir_muncul[d] = 0
    else:
        max_jarak = max(jarak_tercatat) + 1
        rata_jarak = sum(jarak_tercatat) / len(jarak_tercatat)
        for d in terakhir_muncul:
            if terakhir_muncul[d] is None:
                terakhir_muncul[d] = rata_jarak
    bobot = {}
    for d in range(10):
        j = terakhir_muncul[str(d)]
        bobot[str(d)] = 1.0 + (j / max_jarak) * FAKTOR_OVERDUE
    return bobot

def format_hasil(prob):
    """Urutkan probabilitas → ambil 7+1 dan 9 teratas"""
    urut = np.argsort(prob)[::-1].tolist()
    sembilan = urut[:9]
    tujuh = urut[:7] + [urut[9]]  # ← 7 pertama + posisi ke-10
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
    if sampel < 20 + VALIDASI_MIN:
        return None, None, None, None, 0
    X = np.zeros((sampel, LOOKBACK, 4), dtype=np.float32)
    Y = np.zeros((sampel, 4), dtype=np.int32)
    for i in range(sampel):
        X[i] = [dp[j]['angka'] for j in range(i, i + LOOKBACK)]
        Y[i] = dp[i + LOOKBACK]['angka']
    batas = int(0.9 * sampel)
    if (sampel - batas) < VALIDASI_MIN:
        batas = sampel - VALIDASI_MIN
    batas = max(5, batas)
    X_latih = X[:batas]
    Y_latih = [Y[:batas, 0], Y[:batas, 1], Y[:batas, 2], Y[:batas, 3]]
    X_valid = X[batas:]
    Y_valid = [Y[batas:, 0], Y[batas:, 1], Y[batas:, 2], Y[batas:, 3]]
    return X_latih, Y_latih, X_valid, Y_valid, batas

def latih_earlystop(X, Y, Xv, Yv):
    if X is None:
        return None, None
    model = bangun_model()
    # ✅ Patience 10: seimbang antara akurasi & kecepatan
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    hist = model.fit(X, Y, epochs=100, batch_size=32,
                     validation_data=(Xv, Yv), callbacks=[es], verbose=0)
    dipakai = len(hist.history['loss']) - es.patience
    return model, {"epoch": max(5, dipakai), "batch_size": 32, "berhenti_di": len(hist.history['loss'])}

def cari_optuna(X, Y, Xv, Yv):
    if X is None:
        return None, None
    def tujuan(trial):
        epoch = trial.suggest_int('epoch', OPTUNA_EPOCH_MIN, OPTUNA_EPOCH_MAX)
        batch = trial.suggest_categorical('batch_size', OPTUNA_BATCH_CHOICES)
        m = bangun_model()
        # ✅ Patience 6 saat pencarian: cukup cepat & tetap akurat
        es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        h = m.fit(X, Y, epochs=epoch, batch_size=batch,
                  validation_data=(Xv, Yv), callbacks=[es], verbose=0)
        return min(h.history['val_loss'])
    
    study = optuna.create_study(direction='minimize')
    study.optimize(tujuan, n_trials=OPTUNA_CUPIKAN, show_progress_bar=False)
    bp = study.best_params
    
    # ✅ Patience 10 saat latihan akhir: sama dengan EarlyStopping utama
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model = bangun_model()
    model.fit(X, Y, epochs=bp['epoch'], batch_size=bp['batch_size'],
              validation_data=(Xv, Yv), callbacks=[es], verbose=0)
    
    return model, {"epoch": bp['epoch'], "batch_size": bp['batch_size'], "skor_terbaik": round(study.best_value, 6)}

def proses_semua():
    print("📥 Membaca data...")
    req = urllib.request.Request(
        DATA_UNDIAN_URL,
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
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
    dilihat = set()
    bersih = []
    for entri in mentah_data:
        kunci = (entri['pasaran'], entri['tanggal'])
        if kunci not in dilihat:
            dilihat.add(kunci)
            bersih.append(entri)
    mentah_data = bersih
    mentah_data.sort(key=lambda x: (x['tanggal'], x['waktu']))
    data_per_pasaran = {}
    for baris in mentah_data:
        p = baris['pasaran']
        if p not in data_per_pasaran:
            data_per_pasaran[p] = []
        data_per_pasaran[p].append(baris)
    for p in list(data_per_pasaran.keys()):
        if len(data_per_pasaran[p]) > LIMIT_PER_PASARAN:
            data_per_pasaran[p] = data_per_pasaran[p][-LIMIT_PER_PASARAN:]

    # === PILIH PASARAN DI SINI ===
    list_pasaran = sorted(data_per_pasaran.keys())
    # Contoh: list_pasaran = ["HK"]  ← tinggal ubah
    # =============================

    data_terbaru = {p: data_per_pasaran[p][-1] for p in list_pasaran}
    print("\n" + "="*70)
    print("📋 DATA & 2 METODE: EARLY STOPPING | OPTUNA")
    print("="*70)
    for p in list_pasaran:
        d = data_terbaru[p]
        print(f" {p:8} | Terakhir: {d['nomor']} | Total baris: {len(data_per_pasaran[p])}")
    print(f"\n Pengaturan: LOOKBACK={LOOKBACK} | LIMIT={LIMIT_PER_PASARAN}")
    print(f" Optuna: EPOCH {OPTUNA_EPOCH_MIN}-{OPTUNA_EPOCH_MAX} BATCH {OPTUNA_BATCH_CHOICES} | {OPTUNA_CUPIKAN} percobaan")
    print("="*70)

    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "zona_waktu": "WIB / UTC+7",
        "pengaturan": {
            "LOOKBACK": LOOKBACK,
            "LIMIT_PER_PASARAN": LIMIT_PER_PASARAN,
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
        if total < LOOKBACK + 20 + VALIDASI_MIN:
            print(f"\n⚠️ {p:8} — dilewati: butuh minimal {LOOKBACK+20+VALIDASI_MIN} baris, punya {total}")
            continue
        print(f"\n{'─'*70}")
        print(f"🔄 MEMPROSES: {p} | {total} baris")
        print(f"{'─'*70}")
        X, Y, Xv, Yv, batas = siapkan_data(dp)
        if X is None:
            print(f" ⚠️ Data belum cukup untuk dilatih")
            continue
        input_terbaru = np.array([dp[j]['angka'] for j in range(-LOOKBACK, 0)], dtype=np.float32)
        input_terbaru = np.expand_dims(input_terbaru, axis=0)
        hasil_pasaran = {}

        def proses_satu_metode(model, info):
            pred = model.predict(input_terbaru, verbose=0)
            res = {"pengaturan": info}
            for idx, nm in enumerate(nama_posisi):
                prob = pred[idx][0].copy()
                prob /= np.sum(prob)
                bobot = hitung_bobot_overdue(dp, idx)
                prob_ov = prob.copy()
                for d in range(10):
                    prob_ov[d] *= bobot[str(d)]
                prob_ov /= np.sum(prob_ov)
                res[nm] = {"murni": format_hasil(prob), "overdue": format_hasil(prob_ov)}
            return res

        # 1. EARLY STOPPING
        print(f"  ▶️  1/2 EARLY STOPPING...")
        model, info = latih_earlystop(X, Y, Xv, Yv)
        if model is None:
            print(f"     ⚠️ Dilewati — data tidak cukup")
            continue
        hasil_e = proses_satu_metode(model, info)
        hasil_pasaran["early_stopping"] = hasil_e
        print(f"     ✅ Berhenti di epoch {info['berhenti_di']}")

        # 2. OPTUNA
        print(f"  ▶️  2/2 OPTUNA...")
        model, info = cari_optuna(X, Y, Xv, Yv)
        if model is None:
            print(f"     ⚠️ Dilewati — Optuna gagal")
            continue
        hasil_o = proses_satu_metode(model, info)
        hasil_pasaran["optuna"] = hasil_o
        print(f"     ✅ Terbaik: EPOCH={info['epoch']} BATCH={info['batch_size']}")

        hasil_akhir["hasil"][p] = hasil_pasaran
        print(f"\n 📊 CONTOH EKOR +OVERDUE:")
        print(f"    EARLY STOP: 7+1D: {''.join(hasil_e['EKOR']['overdue']['tujuh'])}  9D: {''.join(hasil_e['EKOR']['overdue']['sembilan'])}")
        print(f"    OPTUNA    : 7+1D: {''.join(hasil_o['EKOR']['overdue']['tujuh'])}  9D: {''.join(hasil_o['EKOR']['overdue']['sembilan'])}")

    # ✅ Simpan SEMUA pasaran setelah selesai
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ SEMUA SELESAI → hasil_prediksi.json")
    print(f"{'='*70}")

if __name__ == "__main__":
    proses_semua()
