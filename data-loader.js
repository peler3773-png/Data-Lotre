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
    return pengaturan;
  } catch (err) {
    console.error("❌ Error ambil pengaturan:", err);
    tampilkanPesan("❌ Gagal memuat pengaturan: " + err.message);
    return null;
  }
}

// === BACA FILE DATABASE ===
async function bacaDatabase() {
  if (!pengaturan) {
    tampilkanPesan("⚠️ Pengaturan belum dimuat");
    return [];
  }

  try {
    const urlDB = pengaturan.sumber_database;
    const res = await fetch(urlDB);
    if (!res.ok) throw new Error("File database tidak ditemukan / link mati");

    const teks = await res.text();
    console.log("✅ Database berhasil dibaca");

    // Parse data: pasaran|tanggal|angka
    const baris = teks.trim().split("\n");
    dataHasil = baris.map((b, i) => {
      const kolom = b.split("|");
      return {
        pasaran: kolom[0]?.trim(),
        tanggal: kolom[1]?.trim(),
        angka: kolom[2]?.trim()
      };
    }).filter(r => r.pasaran && r.angka);

    console.log(`✅ ${dataHasil.length} baris data siap`);
    return dataHasil;

  } catch (err) {
    console.error("❌ Error baca database:", err);
    tampilkanPesan("❌ Gagal baca database: " + err.message);
    return [];
  }
}

// === TAMPILKAN PESAN KE HALAMAN ===
function tampilkanPesan(teks) {
  const el = document.getElementById("info-db");
  if (el) el.innerHTML += teks + "<br>";
}

// === TAMPILKAN DATA KE HALAMAN ===
function tampilkanData() {
  const wadah = document.getElementById("daftar-data");
  if (!wadah) return;

  if (!dataHasil.length) {
    wadah.innerHTML = "<p>Belum ada data.</p>";
    return;
  }

  let html = "<h3>📊 Data Hasil Undian</h3><table border='1' cellpadding='6' style='border-collapse:collapse;'>";
  html += "<tr><th>Pasaran</th><th>Tanggal</th><th>Angka</th></tr>";

  dataHasil.forEach(d => {
    html += `<tr><td>${d.pasaran}</td><td>${d.tanggal}</td><td>${d.angka}</td></tr>`;
  });

  html += "</table>";
  wadah.innerHTML = html;
}

// === MULAI SEMUA PROSES ===
async function mulai() {
  tampilkanPesan("🔄 Sedang memuat pengaturan...");
  await ambilPengaturan();
  tampilkanPesan("🔄 Sedang membaca database...");
  await bacaDatabase();
  tampilkanPesan(`✅ Selesai! ${dataHasil.length} data siap.`);
  tampilkanData();
}

// === BUAT TEMPAT TAMPILAN OTOMATIS ===
function buatWadahTampilan() {
  if (document.getElementById("wadah-app")) return;

  const wadah = document.createElement("div");
  wadah.id = "wadah-app";
  wadah.style.padding = "15px";
  wadah.style.fontFamily = "sans-serif";
  wadah.innerHTML = `
    <h2>📡 Data Loader — Lotre</h2>
    <div id="info-db" style="background:#f5f5f5; padding:10px; border-radius:5px; margin-bottom:15px;"></div>
    <div id="daftar-data"></div>
  `;
  document.body.appendChild(wadah);
}

// Jalankan otomatis
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    buatWadahTampilan();
    mulai();
  });
} else {
  buatWadahTampilan();
  mulai();
}