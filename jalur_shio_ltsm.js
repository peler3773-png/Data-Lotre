// jalur_shio_ltsm.js — Sistem Jalur Shio LTSM
// Sumber: pengaturan.json (data_shio + pemetaan_jalur) + data_undian.txt
// Lokasi: Data-Lotre/jalur_shio_ltsm.js
// Update: 2026-10-02

const LTSM_SHIO = {
  pengaturan: null,
  daftarUndian: [],
  
  angkaKeShio: {},
  shioKeAngka: {},
  jalurKeShio: {},
  shioKeJalur: {},

  async muatSemua() {
    try {
      const resPengaturan = await fetch('https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/pengaturan.json');
      if (!resPengaturan.ok) throw new Error('Gagal ambil pengaturan.json');
      this.pengaturan = await resPengaturan.json();
      
      this.bangunPetaShio();
      
      const resUndian = await fetch('https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/data_undian.txt');
      if (!resUndian.ok) throw new Error('Gagal ambil data_undian.txt');
      const teksUndian = await resUndian.text();
      this.uraikanUndian(teksUndian);

      console.log('[LTSM SHIO] ✅ Semua data dimuat');
      return true;
    } catch (err) {
      console.error('[LTSM SHIO] ❌ Gagal muat:', err);
      return false;
    }
  },

  bangunPetaShio() {
    const { data_shio, pemetaan_jalur } = this.pengaturan;

    for (const nama in data_shio) {
      const daftar = data_shio[nama].angka;
      this.shioKeAngka[nama] = daftar;
      daftar.forEach(angka => {
        this.angkaKeShio[angka] = nama;
      });
    }

    for (const kodeJalur in pemetaan_jalur) {
      const daftarShio = pemetaan_jalur[kodeJalur];
      this.jalurKeShio[kodeJalur] = daftarShio;
      daftarShio.forEach(nama => {
        this.shioKeJalur[nama] = kodeJalur;
      });
    }
  },

  uraikanUndian(teks) {
    this.daftarUndian = [];
    const baris = teks.trim().split('\n');
    
    baris.forEach(b => {
      const [pasaran, tgl, nomor4d, waktu] = b.split('|');
      if (!pasaran || !nomor4d || nomor4d.length !== 4) return;
      
      const depan = nomor4d.slice(0, 2);
      const belakang = nomor4d.slice(2, 4);
      
      this.daftarUndian.push({
        pasaran,
        tanggal: tgl,
        nomor4d,
        depan,
        belakang,
        shioDepan: this.angkaKeShio[depan] || null,
        shioBelakang: this.angkaKeShio[belakang] || null,
        jalurDepan: this.angkaKeShio[depan] ? this.shioKeJalur[this.angkaKeShio[depan]] : null,
        jalurBelakang: this.angkaKeShio[belakang] ? this.shioKeJalur[this.angkaKeShio[belakang]] : null,
        waktu
      });
    });

    this.daftarUndian.sort((a, b) => 
      b.tanggal.localeCompare(a.tanggal) || b.waktu.localeCompare(a.waktu)
    );
  },

  ambilTerakhir(pasaran = 'SEMUA', batas = 100) {
    let hasil = this.daftarUndian;
    if (pasaran !== 'SEMUA') {
      hasil = hasil.filter(u => u.pasaran === pasaran);
    }
    return hasil.slice(0, batas);
  },

  hitungJalur(daftarUndian) {
    const rekap = {
      J1: { keluar: 0, angka: [], shio: new Set() },
      J2: { keluar: 0, angka: [], shio: new Set() },
      J3: { keluar: 0, angka: [], shio: new Set() }
    };

    daftarUndian.forEach(u => {
      [u.jalurDepan, u.jalurBelakang].forEach((jalur, idx) => {
        if (!jalur) return;
        const angka = idx === 0 ? u.depan : u.belakang;
        const namaShio = idx === 0 ? u.shioDepan : u.shioBelakang;
        
        rekap[jalur].keluar++;
        rekap[jalur].angka.push(angka);
        rekap[jalur].shio.add(namaShio);
      });
    });

    for (const j in rekap) {
      rekap[j].shio = [...rekap[j].shio];
    }
    return rekap;
  },

  klasifikasiPanas(rekap) {
    const urut = Object.entries(rekap).sort((a, b) => b[1].keluar - a[1].keluar);
    return {
      terpanas: { jalur: urut[0][0], ...urut[0][1] },
      tersedang: { jalur: urut[1][0], ...urut[1][1] },
      terdingin: { jalur: urut[2][0], ...urut[2][1] }
    };
  },

  angkaBelumKeluar(rekap, kodeJalur) {
    const semuaAngka = [];
    const daftarShioJalur = this.jalurKeShio[kodeJalur] || [];
    daftarShioJalur.forEach(nama => {
      semuaAngka.push(...(this.shioKeAngka[nama] || []));
    });
    const sudahKeluar = rekap[kodeJalur]?.angka || [];
    return semuaAngka.filter(a => !sudahKeluar.includes(a));
  },

  daftarPasaran() {
    return [...new Set(this.daftarUndian.map(u => u.pasaran))].sort();
  }
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = LTSM_SHIO;
}
if (typeof window !== 'undefined') {
  window.LTSM_SHIO = LTSM_SHIO;
}
