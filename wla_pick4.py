
# wla_pick4.py — LSTM Murni Per Pasaran
import numpy as np
import pandas as pd
import json
import yaml
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# === Muat Konfigurasi ===
with open("wla_pick4.yml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

np.random.seed(cfg["pengaturan_model"]["acak"])

# === 1. Ambil Pengaturan dari GitHub ===
print("🔄 Memuat pengaturan...")
resp = requests.get(cfg["sumber"]["pengaturan"])
pengaturan = resp.json()
DAFTAR_PASARAN = pengaturan["daftar_pasaran"]
cfg["daftar_pasaran"] = DAFTAR_PASARAN
print(f"✅ Total pasaran terdaftar: {len(DAFTAR_PASARAN)}")

# === 2. Ambil & Parsing Data Undian ===
print("🔄 Memuat data undian...")
resp = requests.get(cfg["sumber"]["data_undian"])
baris = resp.text.strip().split("\n")
data_list = []
for b in baris:
    bagian = b.strip().split("|")
    if len(bagian) >= 4:
        kode, tgl, angka, waktu = bagian[0], bagian[1], bagian[2], bagian[3]
        try:
            angka_int = int(angka)
        except:
            continue
        data_list.append({"pasaran": kode, "tanggal": tgl, "angka": angka_int, "waktu": waktu})

df_all = pd.DataFrame(data_list)
print(f"✅ Data undian dimuat: {len(df_all)} baris")

# === 3. Fungsi Buat Urutan LSTM ===
def buat_urutan(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len, 0])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

# === 4. Proses SATU PASARAN SAJA — Tidak Digabung ===
def proses_pasaran(kode):
    df = df_all[df_all["pasaran"] == kode].copy()
    if len(df) < cfg["pengaturan_model"]["urutan_waktu"] + 5:
        print(f"⚠️ {kode}: Data kurang, dilewati")
        return None

    df = df.sort_values("tanggal").reset_index(drop=True)
    angka = df["angka"].values.reshape(-1, 1)

    # Normalisasi
    scaler = MinMaxScaler(feature_range=(0, 1))
    angka_scaled = scaler.fit_transform(angka)

    # Buat urutan
    X, y = buat_urutan(angka_scaled, cfg["pengaturan_model"]["urutan_waktu"])
    if len(X) == 0:
        print(f"⚠️ {kode}: Urutan gagal dibuat")
        return None

    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["pengaturan_model"]["test_split"], shuffle=False
    )

    # Bangun model KHUSUS pasaran ini
    model = Sequential([
        LSTM(cfg["pengaturan_model"]["unit_lstm1"],
             return_sequences=True,
             input_shape=(cfg["pengaturan_model"]["urutan_waktu"], 1)),
        Dropout(cfg["pengaturan_model"]["dropout"]),
        LSTM(cfg["pengaturan_model"]["unit_lstm2"], return_sequences=False),
        Dropout(cfg["pengaturan_model"]["dropout"]),
        Dense(cfg["pengaturan_model"]["unit_dense"]),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss="mse")

    # Latih
    model.fit(X_train, y_train,
              epochs=cfg["pengaturan_model"]["epoch"],
              batch_size=cfg["pengaturan_model"]["batch_size"],
              validation_data=(X_test, y_test),
              verbose=0)

    # Prediksi berikutnya
    terakhir = angka_scaled[-cfg["pengaturan_model"]["urutan_waktu"]:]
    pred_scaled = model.predict(terakhir.reshape(1, -1, 1), verbose=0)
    prediksi = scaler.inverse_transform(pred_scaled)[0][0]

    # Simpan model
    model.save(f"lstm_{kode.lower()}.h5")
    return {
        "pasaran": kode,
        "nama_panjang": pengaturan["nama_panjang"].get(kode, kode),
        "jumlah_data": len(df),
        "prediksi_bulat": round(prediksi),
        "prediksi_mentah": round(prediksi, 4),
        "model_file": f"lstm_{kode.lower()}.h5"
    }

# === 5. Jalankan Semua Pasaran Aktif ===
hasil_akhir = []
for psr in cfg["pasaran_aktif"]:
    print(f"\n🔄 Memproses: {psr} — {pengaturan['nama_panjang'].get(psr, psr)}")
    res = proses_pasaran(psr)
    if res:
        hasil_akhir.append(res)
        print(f"✅ {psr} → Prediksi: {res['prediksi_bulat']}")

# === 6. Tampilkan Ringkasan ===
print("\n" + "="*60)
print("📋 RINGKASAN PREDIKSI PER PASARAN")
print("="*60)
for h in hasil_akhir:
    print(f"{h['pasaran']:6} | {h['nama_panjang']:25} | Data:{h['jumlah_data']:4} | Prediksi:{h['prediksi_bulat']:02d}")
print("="*60)
