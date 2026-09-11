// =============================================
// KONFIGURASI — PASTIKAN LINK BENAR
// =============================================
const PENGATURAN_URL = "https://raw.githubusercontent.com/peler3773-png/Data-Lotre/main/pengaturan.json";

// =============================================
// VARIABEL GLOBAL
// =============================================
let pengaturan = null;
let dataHasil = [];

// =============================================
// AMBIL PENGATURAN DARI JSON
// =============================================
async function ambilPengaturan() {
  try {
    const res = await fetch(PENGATURAN_URL);
    if (!res.ok) throw new Error("Tidak bisa baca pengaturan.json");
    pengaturan = await res.json();
    console.log("✅ Pengaturan berhasil dimuat");
    console.log("📌 Sumber DB:", pengaturan.sumber_database);
    return pengaturan;
  } catch (err) {
    console.error("❌ Error pengaturan:", err);
    tampilkanPesan("❌ Gagal baca pengaturan: " + err.message);
    return null;
  }
}

// =============================================
// BACA FILE DATABASE
// =============================================
async function bacaDatabase() {
  if (!pengaturan) {
    tampilkanPesan("⚠️ Pengaturan belum siap");
    return [];
  }

  try {
    const res = await fetch(pengaturan.sumber_database);
    if (!res.ok) throw new Error("File database tidak ditemukan");

    const teks = await res.text();
    console.log("✅ Database terbaca");

    // Pisah per baris & per kolom (format: pasaran|tanggal|angka)
    const baris = teks.trim().split("\n");
    dataHasil = baris.map(b => {
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
    console.error("❌ Error database:", err);
    tampilkanPesan("❌ Gagal baca database: " + err.message);
    return [];
  }
}

// =============================================
// TAMPILKAN KE HALAMAN
// =============================================
function tampilkanPesan(teks) {
  const el = document.getElementById("info-app");
  if (el) el.innerHTML += teks + "<br>";
}

function tampilkanTabel() {
  const wadah = document.getElementById("tabel-data");
  if (!wadah) return;

  if (!dataHasil.length) {
    wadah.innerHTML = "<p>Belum ada data.</p>";
    return;
  }

  let html = "<h3>📊 Data Hasil Undian</h3>";
  html += "<table border='1' cellpadding='8' style='border-collapse:collapse;'>";
  html += "<tr style='background:#eee;'><th>Pasaran</th><th>Tanggal</th><th>Angka</th></tr>";

  dataHasil.forEach(d => {
    html += `<tr><td>${d.pasaran}</td><td>${d.tanggal}</td><td>${d.angka}</td></tr>`;
  });

  html += "</table>";
  wadah.innerHTML = html;
}

// =============================================
// JALANKAN SEMUA
// =============================================
async function mulaiSemua() {
  tampilkanPesan("🔄 Membaca pengaturan...");
  await ambilPengaturan();
  
  tampilkanPesan("🔄 Membaca database...");
  await bacaDatabase();
  
  tampilkanPesan(`✅ Selesai! ${dataHasil.length} data dimuat.`);
  tampilkanTabel();
}

// =============================================
// BUAT TEMPAT TAMPILAN OTOMATIS
// =============================================
function buatWadah() {
  if (document.getElementById("wadah-app")) return;

  const wadah = document.createElement("div");
  wadah.id = "wadah-app";
  wadah.style.padding = "20px";
  wadah.style.fontFamily = "Arial, sans-serif";
  wadah.innerHTML = `
    <h2>📡 Data Lotre</h2>
    <div id="info-app" style="background:#f8f8f8; padding:12px; border-radius:6px; margin-bottom:15px;"></div>
    <div id="tabel-data"></div>
  `;
  document.body.appendChild(wadah);
}

// =============================================
// MULAI SAAT HALAMAN SIAP
// =============================================
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    buatWadah();
    mulaiSemua();
  });
} else {
  buatWadah();
  mulaiSemua();
}

// =============================================
// FUNGSI UNTUK DIPAKAI DARI HALAMAN
// =============================================
window.appData = {
  pengaturan: () => pengaturan,
  semuaData: () => dataHasil,
  cari: function(kodePasaran) {
    return dataHasil.filter(d => d.pasaran === kodePasaran);
  },
  mulai: mulaiSemua
};