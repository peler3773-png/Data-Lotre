# jalankan_ai.py — VERSI TELAH DIPERBAIKI & DIUJI
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import urllib.request
import json
from datetime import datetime

# Coba impor TensorFlow dengan penanganan error
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except Exception as e:
    print(f"❌ Gagal memuat TensorFlow: {e}")
    exit(1)

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"
LIMIT_DATA = 80000

# === ATURAN JADWAL LIBUR & LOOKBACK ===
JADWAL_LIBUR = {
    "PS":   {"libur": ["Minggu"],          "lookback": 10},
    "SGP":  {"libur": ["Selasa", "Jumat"], "lookback": 10},
    "TXM":  {"libur": ["Minggu"],          "lookback": 10},
    "TM":   {"libur": ["Minggu"],          "lookback": 10},
    "TMD":  {"libur": ["Senin"],           "lookback": 10},
    "TXD":  {"libur": ["Senin"],           "lookback": 10},
    "TXE":  {"libur": ["Senin"],           "lookback": 10},
    "TXN":  {"libur": ["Senin"],           "lookback": 10},
    "WV":   {"libur": ["Senin"],           "lookback": 10},
    "SCM":  {"libur": ["Minggu"],          "lookback": 10},
}
DEFAULT_LOOKBACK = 10  # Diturunkan agar cepat punya data cukup

def nama_hari(tanggal_str):
    hari_list = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    try:
        dt = datetime.strptime(tanggal_str, "%Y-%m-%d")
        return hari_list[dt.weekday()]
    except:
        return None

def proses_semua():
    print(f"🚀 Mulai: {datetime.now().strftime('%H:%M:%S')}")
    
    # === AMBIL DATA ===
    try:
        req = urllib.request.Request(
            DATA_UNDIAN_URL,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            isi_file = r.read().decode('utf-8')
    except Exception as e:
        print(f"❌ Gagal mengunduh data: {e}")
        return

    mentah_data = []
    semua_pasaran = set()
    for baris in isi_file.strip().splitlines():
        bagian = baris.split('|')
        if len(bagian) < 3:
            continue
        pasaran = bagian[0].strip().upper()
        tanggal = bagian[1].strip()
        angka_4d = bagian[2].strip()
        if len(angka_4d) == 4 and angka_4d.isdigit():
            mentah_data.append({
                "pasaran": pasaran,
                "tanggal": tanggal,
                "angka": [int(d) for d in angka_4d]
            })
            semua_pasaran.add(pasaran)

    # Urutkan berdasarkan tanggal
    mentah_data.sort(key=lambda x: x["tanggal"])

    if len(mentah_data) > LIMIT_DATA:
        mentah_data = mentah_data[-LIMIT_DATA:]

    list_pasaran = sorted(semua_pasaran)
    pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
    total_jenis_pasaran = len(list_pasaran)
    print(f"📊 Ditemukan {total_jenis_pasaran} pasaran, {len(mentah_data)} baris data")

    # === Saring data ===
    data_bersih = {}
    for p in list_pasaran:
        aturan = JADWAL_LIBUR.get(p, {"libur": [], "lookback": DEFAULT_LOOKBACK})
        libur_p = aturan["libur"]
        lb_p = aturan["lookback"]
        daftar = []
        for rec in mentah_data:
            if rec["pasaran"] == p:
                hari = nama_hari(rec["tanggal"])
                if hari and hari not in libur_p:
                    daftar.append(rec)
        data_bersih[p] = {
            "daftar": daftar,
            "lookback": lb_p
        }
        print(f"  {p}: {len(daftar)} rekaman | lookback={lb_p}")

    # === BANGUN MODEL ===
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate

    input_angka = Input(shape=(None, 4), name='input_angka')
    lstm_layer = LSTM(32, activation='relu', return_sequences=False)(input_angka)

    input_pasaran = Input(shape=(1,), name='input_pasaran')
    emb_pasaran = Embedding(input_dim=total_jenis_pasaran, output_dim=4)(input_pasaran)
    flat_pasaran = Flatten()(emb_pasaran)

    gabungan_fitur = concatenate([lstm_layer, flat_pasaran])
    dense_shared = Dense(16, activation='relu')(gabungan_fitur)

    out_as  = Dense(10, activation='softmax', name='output_as')(dense_shared)
    out_kop = Dense(10, activation='softmax', name='output_kop')(dense_shared)
    out_kep = Dense(10, activation='softmax', name='output_kep')(dense_shared)
    out_eko = Dense(10, activation='softmax', name='output_eko')(dense_shared)

    model = Model(inputs=[input_angka, input_pasaran], outputs=[out_as, out_kop, out_kep, out_eko])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # === SUSUN DATA LATIHAN ===
    X_list, Xp_list, Y_as, Y_kop, Y_kep, Y_eko = [], [], [], [], [], []
    for p in list_pasaran:
        df = data_bersih[p]["daftar"]
        lb = data_bersih[p]["lookback"]
        idx = pasaran_ke_idx[p]
        if len(df) < lb + 1:
            print(f"⚠️ {p}: dilewati (butuh {lb+1}, ada {len(df)})")
            continue
        for i in range(len(df) - lb):
            urutan = [df[j]["angka"] for j in range(i, i + lb)]
            target = df[i + lb]["angka"]
            X_list.append(urutan)
            Xp_list.append(idx)
            Y_as.append(target[0])
            Y_kop.append(target[1])
            Y_kep.append(target[2])
            Y_eko.append(target[3])

    if not X_list:
        print("❌ Tidak ada data latihan! Turunkan nilai lookback.")
        return

    X_angka = np.array(X_list, dtype=np.float32)
    X_konteks = np.array(Xp_list, dtype=np.int32)
    y_train = [
        np.array(Y_as),
        np.array(Y_kop),
        np.array(Y_kep),
        np.array(Y_eko)
    ]

    print(f"🧠 Melatih dengan {len(X_list)} sampel...")
    model.fit(
        {'input_angka': X_angka, 'input_pasaran': X_konteks},
        y_train,
        epochs=32,
        batch_size=32,
        verbose=1
    )

    # === PREDIKSI ===
    hasil_akhir = {
        "diperbarui": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_data": len(mentah_data),
        "daftar_pasaran": list_pasaran,
        "hasil": {}
    }
    nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]

    for p in list_pasaran:
        df = data_bersih[p]["daftar"]
        lb = data_bersih[p]["lookback"]
        idx_target = np.array([pasaran_ke_idx[p]])

        if len(df) < lb:
            hasil_akhir["hasil"][p] = {"keterangan": "Data belum cukup", "lookback": lb}
            continue

        urutan_terbaru = np.array(
            [rec["angka"] for rec in df[-lb:]],
            dtype=np.float32
        )
        input_terbaru = np.expand_dims(urutan_terbaru, axis=0)
        pred = model.predict(
            {'input_angka': input_terbaru, 'input_pasaran': idx_target},
            verbose=0
        )

        posisi_data = {}
        for idx_pos, nama in enumerate(nama_posisi):
            urut = np.argsort(-pred[idx_pos][0]).tolist()
            posisi_data[nama] = {
                "lima": [str(a) for a in urut[:5]],
                "sembilan": [str(a) for a in urut[:9]]
            }
        posisi_data["lookback_digunakan"] = lb
        hasil_akhir["hasil"][p] = posisi_data

    # === SIMPAN ===
    with open("hasil_prediksi.json", "w", encoding="utf-8") as f:
        json.dump(hasil_akhir, f, ensure_ascii=False, indent=2)

    print(f"✅ Selesai → hasil_prediksi.json")

if __name__ == "__main__":
    proses_semua()
