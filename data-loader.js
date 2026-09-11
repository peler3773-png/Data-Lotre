// === KONFIGURASI ===
const PENGATURAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/pengaturan.json";

// === VARIABEL GLOBAL ===
let pengaturan = null;
let dataHasil = [];

// === AMBIL PENGATURAN DARI JSON ===
async function ambilPengaturan() {
  try {
    const res = await fetch(PENGATURAN_URL);
    if (!res.ok) throw new Error("Gagal baca pengaturan.json");
    pengaturan = await res.json();
    console.log("✅ Pengaturan berhasil dimuat");
    console.log("📌 Sumber DB:", pengaturan.sumber_database);
    console.log("📌 Nama Tabel:", pengaturan.nama_tabel);
    return pengaturan;
  } catch (err) {
    console.error("❌ Error ambil pengaturan:", err);
    return null;
  }
}

// === BACA FILE DATABASE (.db / teks) ===
async function bacaDatabase() {
  if (!pengaturan) {
    console.warn("⚠️ Pengaturan belum dimuat");
    return [];
  }

  try {
    const urlDB = pengaturan.sumber_database;
    const res = await fetch(urlDB);
    if (!res.ok) throw new Error("File database tidak ditemukan / link mati");

    // Baca sebagai teks
    const teks = await res.text();
    console.log("✅ Database berhasil dibaca, ukuran:", teks.length, "karakter");

    // === PARSE DATA (sesuaikan format isi .db kamu) ===
    // Asumsi tiap baris: kode_pasaran|tanggal|angka
    const baris = teks.trim().split("\n");
    dataHasil = baris.map((b, i) => {
      const kolom = b.split("|");
      return {
        pasaran: kolom[0]?.trim(),
        tanggal: kolom[1]?.trim(),
        angka: kolom[2]?.trim(),
        barisKe: i + 1
      };
    }).filter(r => r.pasaran && r.angka);

    console.log(`✅ Berhasil parse ${dataHasil.length} baris data`);
    return dataHasil;

  } catch (err) {
    console.error("❌ Error baca database:", err);
    return [];
  }
}

// === AMBIL DATA BERDASARKAN KODE PASARAN ===
function cariDataPasaran(kodePasaran) {
  if (!dataHasil.length) return [];
  return dataHasil.filter(d => d.pasaran === kodePasaran);
}

// === JALANKAN OTOMATIS ===
async function mulai() {
  console.log("🔄 Memulai proses...");
  await ambilPengaturan();
  await bacaDatabase();
  console.log("✅ Selesai! Data siap dipakai.");
}

// === EKSPOR UNTUK DIPAKAI DI HALAMAN ===
window.appData = {
  pengaturan: () => pengaturan,
  semuaData: () => dataHasil,
  cari: cariDataPasaran,
  mulai
};

// Jalankan saat halaman siap
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", mulai);
} else {
  mulai();
}
