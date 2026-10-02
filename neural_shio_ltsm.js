// NEURAL NETWORK — JALUR SHIO LTSM
// Sumber: pengaturan.json + data_undian.txt
// Tipe: Jaringan Feedforward 1 Lapisan Tersembunyi
// Lokasi: Data-Lotre/neural_shio_ltsm.js
// Update: 2026-10-02

const NEURAL_SHIO = {
  pengaturan: null,
  daftarUndian: [],
  
  angkaKeShio: {},
  shioKeJalur: {},
  jalurKeIndeks: { J1: 0, J2: 1, J3: 2 },
  indeksKeJalur: ['J1', 'J2', 'J3'],

  jaringan: {
    ukuranInput: 12,
    ukuranTersembunyi: 8,
    ukuranOutput: 3,
    bobotInput: [],
    bobotOutput: [],
    biasTersembunyi: [],
    biasOutput: [],
    lajuBelajar: 0.08,
    terlatih: false
  },

  riwayat: { akurasi: [], kerugian: [] },

  async muatSemua() {
    try {
      const resPengaturan = await fetch('https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/pengaturan.json');
      this.pengaturan = await resPengaturan.json();
      
      this.bangunPeta();

      const resUndian = await fetch('https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt');
      const teksUndian = await resUndian.text();
      this.uraikanUndian(teksUndian);

      this.inisialisasiBobot();
      console.log('[NEURAL SHIO] ✅ Data dimuat — Siap latih');
      return true;
    } catch (err) {
      console.error('[NEURAL SHIO] ❌ Gagal:', err);
      return false;
    }
  },

  bangunPeta() {
    const { data_shio, pemetaan_jalur } = this.pengaturan;
    
    for (const nama in data_shio) {
      data_shio[nama].angka.forEach(angka => {
        this.angkaKeShio[angka] = nama;
      });
    }

    for (const jalur in pemetaan_jalur) {
      pemetaan_jalur[jalur].forEach(nama => {
        this.shioKeJalur[nama] = jalur;
      });
    }
  },

  uraikanUndian(teks) {
    this.daftarUndian = [];
    teks.trim().split('\n').forEach(b => {
      const [pasaran, tgl, nomor4d, waktu] = b.split('|');
      if (!nomor4d || nomor4d.length !== 4) return;
      
      const depan = nomor4d.slice(0, 2);
      const belakang = nomor4d.slice(2, 4);
      const shioDepan = this.angkaKeShio[depan];
      const shioBelakang = this.angkaKeShio[belakang];

      this.daftarUndian.push({
        pasaran, tanggal: tgl, nomor4d, depan, belakang,
        shioDepan, shioBelakang,
        jalurDepan: shioDepan ? this.shioKeJalur[shioDepan] : null,
        jalurBelakang: shioBelakang ? this.shioKeJalur[shioBelakang] : null,
        waktu
      });
    });
    this.daftarUndian.sort((a, b) => 
      b.tanggal.localeCompare(a.tanggal) || b.waktu.localeCompare(a.waktu)
    );
  },

  inisialisasiBobot() {
    const j = this.jaringan;
    const acak = () => (Math.random() - 0.5) * 0.4;

    j.bobotInput = Array(j.ukuranTersembunyi).fill(0).map(() =>
      Array(j.ukuranInput).fill(0).map(acak)
    );
    j.bobotOutput = Array(j.ukuranOutput).fill(0).map(() =>
      Array(j.ukuranTersembunyi).fill(0).map(acak)
    );
    j.biasTersembunyi = Array(j.ukuranTersembunyi).fill(0).map(acak);
    j.biasOutput = Array(j.ukuranOutput).fill(0).map(acak);
    j.terlatih = false;
  },

  sigmoid(x) { return 1 / (1 + Math.exp(-x)); },
  sigmoidTurunan(y) { return y * (1 - y); },
  
  softmax(arr) {
    const max = Math.max(...arr);
    const exp = arr.map(v => Math.exp(v - max));
    const jumlah = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / jumlah);
  },

  siapkanData(pasaran = 'SGP', panjangJendela = 6) {
    const data = this.daftarUndian.filter(u => 
      u.pasaran === pasaran && u.jalurDepan && u.jalurBelakang
    ).reverse();

    const masukan = [];
    const sasaran = [];

    for (let i = panjangJendela; i < data.length; i++) {
      const jendela = data.slice(i - panjangJendela, i);
      const target = data[i];

      const vektorInput = Array(12).fill(0);
      jendela.forEach((item, idx) => {
        const pos = (idx % 6) * 2;
        const urutanDepan = this.pengaturan.data_shio[item.shioDepan]?.urutan || 0;
        const urutanBelakang = this.pengaturan.data_shio[item.shioBelakang]?.urutan || 0;
        if (urutanDepan) vektorInput[pos] = urutanDepan / 12;
        if (urutanBelakang) vektorInput[pos + 1] = urutanBelakang / 12;
      });

      const vektorTarget = [0, 0, 0];
      const idxTarget = this.jalurKeIndeks[target.jalurDepan];
      if (idxTarget !== undefined) vektorTarget[idxTarget] = 1;

      masukan.push(vektorInput);
      sasaran.push(vektorTarget);
    }

    return { masukan, sasaran };
  },

  maju(vektorInput) {
    const j = this.jaringan;

    const nilaiTersembunyi = [];
    for (let h = 0; h < j.ukuranTersembunyi; h++) {
      let jumlah = j.biasTersembunyi[h];
      for (let i = 0; i < j.ukuranInput; i++) {
        jumlah += vektorInput[i] * j.bobotInput[h][i];
      }
      nilaiTersembunyi.push(this.sigmoid(jumlah));
    }

    const nilaiOutput = [];
    for (let o = 0; o < j.ukuranOutput; o++) {
      let jumlah = j.biasOutput[o];
      for (let h = 0; h < j.ukuranTersembunyi; h++) {
        jumlah += nilaiTersembunyi[h] * j.bobotOutput[o][h];
      }
      nilaiOutput.push(jumlah);
    }

    return {
      tersembunyi: nilaiTersembunyi,
      output: this.softmax(nilaiOutput)
    };
  },

  latih(pasaran = 'SGP', epoch = 800) {
    const { masukan, sasaran } = this.siapkanData(pasaran);
    if (!masukan.length) {
      console.warn('[NEURAL SHIO] Tidak ada data latih');
      return false;
    }

    this.riwayat = { akurasi: [], kerugian: [] };
    const j = this.jaringan;

    for (let e = 0; e < epoch; e++) {
      let totalSalah = 0;
      let benar = 0;

      for (let i = 0; i < masukan.length; i++) {
        const hasil = this.maju(masukan[i]);
        const prediksi = hasil.output;
        const tersembunyi = hasil.tersembunyi;

        const selisihOutput = [];
        for (let o = 0; o < j.ukuranOutput; o++) {
          selisihOutput[o] = sasaran[i][o] - prediksi[o];
          totalSalah += Math.abs(selisihOutput[o]);
        }
        if (prediksi.indexOf(Math.max(...prediksi)) === sasaran[i].indexOf(1)) benar++;

        const deltaOutput = [...selisihOutput];
        for (let o = 0; o < j.ukuranOutput; o++) {
          for (let h = 0; h < j.ukuranTersembunyi; h++) {
            j.bobotOutput[o][h] += deltaOutput[o] * tersembunyi[h] * j.lajuBelajar;
          }
          j.biasOutput[o] += deltaOutput[o] * j.lajuBelajar;
        }

        const deltaTersembunyi = Array(j.ukuranTersembunyi).fill(0);
        for (let h = 0; h < j.ukuranTersembunyi; h++) {
          let akumulasi = 0;
          for (let o = 0; o < j.ukuranOutput; o++) {
            akumulasi += selisihOutput[o] * j.bobotOutput[o][h];
          }
          deltaTersembunyi[h] = akumulasi * this.sigmoidTurunan(tersembunyi[h]);
        }

        for (let h = 0; h < j.ukuranTersembunyi; h++) {
          for (let ii = 0; ii < j.ukuranInput; ii++) {
            j.bobotInput[h][ii] += deltaTersembunyi[h] * masukan[i][ii] * j.lajuBelajar;
          }
          j.biasTersembunyi[h] += deltaTersembunyi[h] * j.lajuBelajar;
        }
      }

      if ((e + 1) % 50 === 0) {
        const akurasi = (benar / masukan.length * 100).toFixed(1);
        const rugi = (totalSalah / masukan.length).toFixed(4);
        this.riwayat.akurasi.push(akurasi);
        this.riwayat.kerugian.push(rugi);
        console.log(`[NEURAL SHIO] Epoch ${e+1}/${epoch} → Akurasi: ${akurasi}% | Rugi: ${rugi}`);
      }
    }

    j.terlatih = true;
    console.log('[NEURAL SHIO] ✅ Pelatihan selesai');
    return true;
  },

  prediksiBerikutnya(pasaran = 'SGP', panjangJendela = 6) {
    if (!this.jaringan.terlatih) {
      console.warn('[NEURAL SHIO] Belum dilatih — jalankan .latih() dulu');
      return null;
    }

    const data = this.daftarUndian.filter(u => 
      u.pasaran === pasaran && u.jalurDepan && u.jalurBelakang
    );
    if (data.length < panjangJendela) return null;

    const jendela = data.slice(0, panjangJendela).reverse();
    const vektorInput = Array(12).fill(0);
    jendela.forEach((item, idx) => {
      const pos = (idx % 6) * 2;
      const urutanDepan = this.pengaturan.data_shio[item.shioDepan]?.urutan || 0;
      const urutanBelakang = this.pengaturan.data_shio[item.shioBelakang]?.urutan || 0;
      if (urutanDepan) vektorInput[pos] = urutanDepan / 12;
      if (urutanBelakang) vektorInput[pos + 1] = urutanBelakang / 12;
    });

    const hasil = this.maju(vektorInput);
    const peluang = hasil.output;

    const urut = peluang.map((p, i) => ({
      jalur: this.indeksKeJalur[i],
      peluang: (p * 100).toFixed(1) + '%',
      nilai: p
    })).sort((a, b) => b.nilai - a.nilai);

    return {
      teratas: urut[0].jalur,
      peluangTeratas: urut[0].peluang,
      urutan: urut,
      jendelaTerakhir: jendela.map(d => `${d.depan}${d.belakang} → ${d.jalurDepan} / ${d.jalurBelakang}`)
    };
  },

  async jalankanLengkap(pasaran = 'SGP') {
    const siap = await this.muatSemua();
    if (!siap) return { ok: false };

    await this.latih(pasaran, 800);
    const prediksi = this.prediksiBerikutnya(pasaran);

    return {
      ok: true,
      pasaran,
      prediksi,
      akurasiTerakhir: this.riwayat.akurasi.at(-1),
      jumlahData: this.daftarUndian.filter(d => d.pasaran === pasaran).length
    };
  }
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = NEURAL_SHIO;
}
if (typeof window !== 'undefined') {
  window.NEURAL_SHIO = NEURAL_SHIO;
}
