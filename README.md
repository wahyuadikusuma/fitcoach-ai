# 🏋️‍♂️ FitCoach AI - Asisten Gym Personal & Perancang Program Latihan Cerdas

> **Final Project: LLM-Based Tools and Gemini API Integration for Data Scientists**

FitCoach AI adalah asisten gym dan kebugaran personal berbasis kecerdasan buatan (*Artificial Intelligence*) yang didukung oleh **Google Gemini API** dan antarmuka web modern **Streamlit**. Aplikasi ini dirancang untuk mendampingi pengguna dan klien gym dalam mencapai target kebugaran secara ilmiah, aman, dan terpersonalisasi.

Sebelum memberikan rekomendasi program latihan atau pola makan, FitCoach AI secara proaktif melakukan **profiling biometrik klien** (**Jenis Kelamin, Usia, Tinggi Badan, dan Berat Badan**) untuk mengkalkulasi komposisi tubuh serta kebutuhan energi harian secara akurat.

---

## 📸 Tangkapan Layar Antarmuka (UI Screenshots)

| Overview Aplikasi FitCoach AI | Profiling Biometrik & Form Data Tubuh |
| :---: | :---: |
| ![FitCoach UI Overview](screenshots/fitcoach_ui_overview.png) | ![FitCoach Profiling Sidebar](screenshots/fitcoach_sidebar_profiling.png) |

| Kartu Metrik Biometrik (BMI & Kalori) | Respon Program Latihan (Gemini 3.6 Flash) |
| :---: | :---: |
| ![FitCoach Biometrics](screenshots/fitcoach_biometrics.png) | ![FitCoach Chat Response](screenshots/fitcoach_chat_response.png) |

---

## 🎯 Use Case & Alur Kerja

1. **Onboarding & Profiling Klien (Wajib)**:
   - Saat pertama kali berinteraksi, asisten meminta 4 data fisik dasar: **Jenis Kelamin**, **Usia**, **Tinggi Badan (cm)**, dan **Berat Badan (kg)**.
   - Pengguna dapat mengisi data melalui form interaktif di sidebar maupun mengetik langsung di chat (didukung *auto-biometric entity extraction*).
2. **Kalkulator Biometrik Medis Otomatis**:
   - **BMI (Body Mass Index)**: Dihitung dengan rumus $BB / (TB/100)^2$ disertai status kategori (Underweight, Normal, Overweight, Obese) dan rekomendasi kesehatan.
   - **BMR (Basal Metabolic Rate)**: Dihitung menggunakan persamaan medis *Mifflin-St Jeor* berdasarkan jenis kelamin, berat, tinggi, dan usia.
   - **TDEE (Total Daily Energy Expenditure)**: Estimasi pengeluaran energi harian berdasarkan tingkat aktivitas fisik (Sedentary hingga Extremely Active).
   - **Kebutuhan Kalori & Makronutrien**: Perhitungan otomatis target kalori (defisit 20% untuk fat loss, surplus 12% untuk bulking) serta gram Protein, Karbohidrat, dan Lemak.
3. **Penyusunan Program Latihan Terstruktur**:
   - Push / Pull / Legs (PPL) Split
   - Upper / Lower Body Split
   - Full Body Workout (3-4 hari)
   - Home Workout (Tanpa alat / Bodyweight & Calisthenics)
   - Panduan pemanasan dinamis, mobilitas sendi, serta pencegahan cedera (*form over weight*).

---

## 🎛️ Parameter Kreatif (Sesuai Rubrik Tugas)

Sesuai dengan kriteria tugas akhir, FitCoach AI menerapkan beberapa parameter kreatif:

1. **Pilihan Persona Pelatih (Gaya Bahasa & Karakter)**:
   - 🔥 **Coach Alex (Motivator Energik)**: Sangat bersemangat, antusias, memotivasi mental juara ("Gaspol! Jangan kasih kendor!").
   - ⚡ **Coach Viktor (Drill Sergeant)**: Tegas, disiplin tinggi, militeristik, tanpa alasan ("No excuses!").
   - 🧬 **Dr. Bryan, CSCS (Sports Scientist)**: Ilmiah, analitis, berbasis bukti riset fisiologi, RPE, dan biomekanika sendi.
   - 🌱 **Coach Maya (Empathetic & Friendly)**: Ramah, sabar, empatik, cocok untuk pemula yang baru pertama kali ke gym.
2. **Model LLM Selector**:
   - Mendukung model Gemini resmi terbaru: `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.5-flash`, dan `gemini-flash-latest`.
3. **Kreativitas Respon (Temperature Slider)**:
   - Slider temperatur (0.0 – 1.0) untuk mengatur tingkat deterministik vs variatif respon pelatih.
4. **Quick Action Prompt Presets**:
   - Tombol instan 1-klik untuk meminta program PPL 4 hari, perhitungan makro & menu makan, home workout, atau panduan pemanasan.
5. **Memory & Export Program**:
   - Riwayat percakapan disimpan dalam sesi aktif (`st.session_state`).
   - Tombol **Download .md** untuk mengekspor program latihan dan riwayat konsultasi ke file Markdown.

---

## 🏗️ Struktur Direktori Proyek

```text
final-project-llm/
├── app.py                      # File utama aplikasi Streamlit (UI & alur chat)
├── requirements.txt            # Daftar pustaka Python yang dibutuhkan
├── README.md                   # Dokumentasi lengkap proyek
├── .env.example                # Template konfigurasi API Key
├── .gitignore                  # File pengabaian Git (venv, .env, cache, dll)
├── screenshots/                # Tangkapan layar User Interface untuk deliverable
│   ├── fitcoach_ui_overview.png
│   ├── fitcoach_sidebar_profiling.png
│   ├── fitcoach_biometrics.png
│   ├── fitcoach_sidebar_dropdown.png
│   └── fitcoach_chat_response.png
└── utils/                      # Modul logika dan utilitas
    ├── fitness_calc.py         # Rumus BMI, BMR (Mifflin-St Jeor), TDEE, & Makro
    └── gemini_client.py        # Integrasi Gemini API, System Instruction, & Streaming
```

---

## 🚀 Panduan Instalasi & Menjalankan Aplikasi

### 1. Clone Repositori
```bash
git clone <URL_REPOSITORI_ANDA>
cd final-project-llm
```

### 2. Buat & Aktifkan Virtual Environment
```bash
python3 -m venv venv

# Di macOS / Linux:
source venv/bin/activate

# Di Windows:
venv\Scripts\activate
```

### 3. Instal Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Konfigurasi Google Gemini API Key
Anda dapat mengatur API Key dengan salah satu dari dua cara:
1. **Melalui File `.env`**:
   Salin `.env.example` menjadi `.env` lalu masukkan API key Anda:
   ```bash
   cp .env.example .env
   ```
   Isi `.env`:
   ```env
   GEMINI_API_KEY=AIzaSy...
   ```
2. **Langsung pada Antarmuka Web**:
   Masukkan Gemini API Key pada kolom *sidebar* aplikasi saat pertama kali dibuka.
   *(Dapatkan API key gratis di [Google AI Studio](https://aistudio.google.com/app/apikey))*.

### 5. Jalankan Aplikasi
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di browser Anda pada alamat: `http://localhost:8501`.

---

## 👨‍💻 Kontributor
- **Nama Peserta**: Wahyu
- **Kelas / Program**: LLM-Based Tools and Gemini API Integration for Data Scientists
- **Proyek**: Final Project Chatbot Gym Personal (FitCoach AI)
