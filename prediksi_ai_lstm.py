# prediksi_ai_lstm.py — DEEP LEARNING CROSS-MARKET AI
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding, Flatten, concatenate

tf.get_logger().setLevel('ERROR')

DATA_UNDIAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt"

def jalankan_prediksi(pasaran_target="NYM"):
    try:
        req = urllib.request.Request(DATA_UNDIAN_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            isi_file = response.read().decode('utf-8')
        
        mentah_data = []
        semua_pasaran = set()
        for baris in isi_file.strip().split('\n'):
            if not baris.strip(): continue
            bagian = baris.split('|')
            if len(bagian) >= 3:
                pasaran = bagian[0].strip().upper()
                angka_4d = bagian[2].strip()
                if len(angka_4d) == 4 and angka_4d.isdigit():
                    mentah_data.append({"pasaran": pasaran, "angka": [int(d) for d in angka_4d]})
                    semua_pasaran.add(pasaran)
        
        LIMIT_DATA = 25000
        if len(mentah_data) > LIMIT_DATA:
            mentah_data = mentah_data[-LIMIT_DATA:]
        
        list_pasaran = sorted(list(semua_pasaran))
        pasaran_ke_idx = {p: i for i, p in enumerate(list_pasaran)}
        total_jenis_pasaran = len(list_pasaran)
        
        if pasaran_target not in pasaran_ke_idx:
            return f"❌ Pasaran '{pasaran_target}' tidak ditemukan!"
        
        lookback = 10
        X_angka, X_konteks, Y = [], [], []
        for i in range(len(mentah_data) - lookback):
            X_angka.append([d["angka"] for d in mentah_data[i:i+lookback]])
            X_konteks.append(pasaran_ke_idx[mentah_data[i+lookback]["pasaran"]])
            Y.append(mentah_data[i+lookback]["angka"])
        
        X_angka = np.array(X_angka)
        X_konteks = np.array(X_konteks)
        Y = np.array(Y)
        
        # Bangun Model LSTM
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
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        y_train = [Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]]
        model.fit({'input_angka': X_angka, 'input_pasaran': X_konteks}, y_train, epochs=45, batch_size=128, verbose=0)
        
        # Prediksi
        input_terbaru = np.array([d["angka"] for d in mentah_data[-lookback:]])
        input_terbaru = np.expand_dims(input_terbaru, axis=0)
        idx_target = np.array([pasaran_ke_idx[pasaran_target]])
        pred = model.predict({'input_angka': input_terbaru, 'input_pasaran': idx_target}, verbose=0)
        
        nama_posisi = ["AS", "KOP", "KEPALA", "EKOR"]
        hasil = []
        hasil.append(f"--- DEEP LEARNING LSTM — TARGET: {pasaran_target} ---")
        hasil.append(f"Total Data: {len(mentah_data)} periode | Pasaran: {total_jenis_pasaran} jenis")
        hasil.append("-" * 50)
        
        for pos in range(4):
            urut = np.argsort(pred[pos][0])[::-1]
            lima = sorted([str(a) for a in urut[:5]])
            top9 = [str(a) for a in urut[:9]]
            hasil.append(f"📍 {nama_posisi[pos]}:")
            hasil.append(f" 👉 [ {''.join(lima)} ]")
            hasil.append(f" 💡 Top 9: {''.join(top9)}")
            hasil.append("")
        
        return "\n".join(hasil)
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    print(jalankan_prediksi("NYM"))
