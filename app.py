"""
FitCoach AI - Personal Gym Assistant & Workout Planner Chatbot
Didukung oleh Google Gemini API & Streamlit
Tugas Akhir: LLM-Based Tools and Gemini API Integration for Data Scientists
"""

import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from utils.fitness_calc import (
    calculate_bmi,
    get_bmi_category,
    calculate_bmr,
    calculate_tdee,
    calculate_target_nutrition,
    generate_client_context_summary,
    ACTIVITY_MULTIPLIERS,
)
from utils.gemini_client import (
    COACH_PERSONAS,
    AVAILABLE_MODELS,
    build_system_instruction,
    stream_gemini_response,
)

# Muat environment variables dari .env jika ada
load_dotenv()

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="FitCoach AI - Asisten Gym Personal",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS: MODERN DARK GYM AESTHETIC
# ==========================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F1F5F9;
}

h1, h2, h3, h4, h5, .hero-title {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.01em;
}

/* Background canvas */
.stApp {
    background-color: #0B0F19;
}

/* Sidebar clean container & widened layout */
[data-testid="stSidebar"] {
    background-color: #0F1626;
    border-right: 1px solid #1E293B;
    min-width: 440px !important;
    max-width: 500px !important;
    width: 460px !important;
}

[data-testid="stSidebar"] > div:first-child {
    min-width: 440px !important;
    max-width: 500px !important;
    width: 460px !important;
}

/* Ensure selectbox selected text and dropdown options never truncate */
div[data-baseweb="select"] {
    width: 100% !important;
}

div[data-baseweb="select"] * {
    white-space: normal !important;
    text-overflow: clip !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
}

/* Dropdown popover menu styling */
div[data-baseweb="popover"],
ul[data-testid="stSelectboxVirtualDropdown"],
div[role="listbox"] {
    max-width: 480px !important;
}

ul[data-testid="stSelectboxVirtualDropdown"] li,
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li,
div[role="listbox"] div[role="option"],
div[role="listbox"] li {
    white-space: normal !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
    line-height: 1.4 !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
    height: auto !important;
    min-height: 40px !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
    color: #F8FAFC;
}

/* Hero card modern */
.hero-card {
    background: #111A2E;
    border: 1px solid #1E2E4A;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(0, 230, 118, 0.12);
    border: 1px solid rgba(0, 230, 118, 0.4);
    border-radius: 20px;
    color: #00E676;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0 0 0.4rem 0;
}

.hero-title span {
    color: #00E676;
}

.hero-subtitle {
    color: #94A3B8;
    font-size: 0.92rem;
    line-height: 1.5;
    margin: 0;
}

.profile-pill {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 0.84rem;
    font-weight: 600;
}

/* Metric boxes */
.metric-box {
    background: #131E33;
    border: 1px solid #233352;
    border-radius: 10px;
    padding: 0.85rem 0.6rem;
    text-align: center;
}

.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    color: #94A3B8;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}

.metric-val {
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 3px 0;
}

.metric-sub {
    font-size: 0.72rem;
    font-weight: 600;
}

/* Chat container & bubbles */
[data-testid="stChatMessage"] {
    background-color: #121A2B;
    border: 1px solid #1E2B42;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.85rem;
}

/* Form inputs unified styling */
.stTextInput input, .stNumberInput input {
    background-color: #151F33 !important;
    border: 1px solid #263857 !important;
    color: #F8FAFC !important;
    border-radius: 8px !important;
}

.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #00E676 !important;
    box-shadow: 0 0 0 1px #00E676 !important;
}

/* Expander styling */
[data-testid="stExpander"] {
    background-color: #121A2B !important;
    border: 1px solid #202D45 !important;
    border-radius: 10px !important;
    margin-bottom: 1rem !important;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: #00E676 !important;
    color: #0B0F19 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    transition: transform 0.15s ease, opacity 0.15s ease !important;
}

.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Secondary / regular buttons */
.stButton > button[kind="secondary"] {
    background-color: #151F33 !important;
    border: 1px solid #283A59 !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: #00E676 !important;
    color: #00E676 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "profile" not in st.session_state:
    st.session_state.profile = {
        "profile_complete": False,
        "gender": "Pria",
        "age": 25,
        "height": 172.0,
        "weight": 68.0,
        "activity_level": "Moderately Active (Olahraga Sedang 3-5 hari/minggu)",
        "goal": "Pembentukan Otot (Hypertrophy)",
        "experience": "Pemula (Beginner)",
        "location": "Gym Komersial (Alat Lengkap)",
    }

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")


# ==========================================
# HELPER: PARSING BIOMETRIK DARI CHAT
# ==========================================
def try_parse_biometrics_from_text(text: str) -> bool:
    """Mencoba mengekstrak data usia, TB, BB, dan gender dari input pengguna jika ada."""
    updated = False
    lower_text = text.lower()

    # Ekstrak Gender
    if any(k in lower_text for k in ["pria", "laki-laki", "cowok", "pria dewasa", "male"]):
        st.session_state.profile["gender"] = "Pria"
        updated = True
    elif any(k in lower_text for k in ["wanita", "perempuan", "cewek", "female"]):
        st.session_state.profile["gender"] = "Wanita"
        updated = True

    # Ekstrak Usia (contoh: "usia 25", "umur 25 tahun", "25 thn")
    age_match = re.search(r'(?:usia|umur)\s*(?:adalah|:)?\s*(\d{2})', lower_text)
    if not age_match:
        age_match = re.search(r'(\d{2})\s*(?:tahun|thn)', lower_text)
    if age_match:
        age_val = int(age_match.group(1))
        if 12 <= age_val <= 90:
            st.session_state.profile["age"] = age_val
            updated = True

    # Ekstrak Tinggi Badan (contoh: "tb 175", "tinggi 175 cm", "175cm")
    tb_match = re.search(r'(?:tinggi\s*badan|tinggi|tb)\s*(?:adalah|:)?\s*(\d{2,3})', lower_text)
    if not tb_match:
        tb_match = re.search(r'(\d{3})\s*(?:cm|sentimeter)', lower_text)
    if tb_match:
        tb_val = float(tb_match.group(1))
        if 100 <= tb_val <= 230:
            st.session_state.profile["height"] = tb_val
            updated = True

    # Ekstrak Berat Badan (contoh: "bb 70", "berat 70 kg", "70kg")
    bb_match = re.search(r'(?:berat\s*badan|berat|bb)\s*(?:adalah|:)?\s*(\d{2,3}(?:\.\d)?)', lower_text)
    if not bb_match:
        bb_match = re.search(r'(\d{2,3}(?:\.\d)?)\s*(?:kg|kilo)', lower_text)
    if bb_match:
        bb_val = float(bb_match.group(1))
        if 30 <= bb_val <= 250:
            st.session_state.profile["weight"] = bb_val
            updated = True

    # Jika semua variabel penting sudah realistis, tandai lengkap
    p = st.session_state.profile
    if p["age"] > 0 and p["height"] > 0 and p["weight"] > 0:
        st.session_state.profile["profile_complete"] = True

    return updated


# ==========================================
# SIDEBAR: PENGATURAN & PROFIL KLIEN
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi Sistem")

    # Input API Key
    api_key_input = st.text_input(
        "🔑 Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Dapatkan API Key gratis di Google AI Studio (https://aistudio.google.com/app/apikey)",
        placeholder="Tempel Google Gemini API Key di sini (contoh: AIzaSy...)",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input

    if not st.session_state.api_key:
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: -6px; margin-bottom: 8px;">
                💡 Belum punya API key? <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #00E676; text-decoration: underline;">Buat gratis di Google AI Studio</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Parameter Kreatif AI
    st.markdown("### 🎭 Persona & Model AI")
    selected_persona = st.selectbox(
        "Pilih Karakter Pelatih (Persona):",
        options=list(COACH_PERSONAS.keys()),
        index=0,
        help="Gaya bahasa dan pendekatan pelatih saat menjawab pertanyaan Anda.",
    )

    selected_model = st.selectbox(
        "Pilih Model Gemini:",
        options=AVAILABLE_MODELS,
        index=0,
        help="Model Gemini yang digunakan untuk pemrosesan LLM.",
    )

    temperature = st.slider(
        "Kreativitas Respon (Temperature):",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Nilai lebih rendah = lebih presisi/konservatif; nilai lebih tinggi = lebih variatif.",
    )

    st.markdown("---")

    # Form Profiling Klien (Wajib: Jenis Kelamin, Usia, TB, BB)
    st.markdown("### 📋 Profil Fisik Klien")
    st.caption("Data ini digunakan AI untuk mengkalkulasi BMI, BMR, dan program latihan yang presisi.")

    profile = st.session_state.profile

    with st.expander("📝 Formulir Data Tubuh", expanded=not profile["profile_complete"]):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"], index=0 if profile["gender"] == "Pria" else 1)
        with col_g2:
            age = st.number_input("Usia (Tahun)", min_value=12, max_value=90, value=int(profile["age"]), step=1)

        col_h1, col_w1 = st.columns(2)
        with col_h1:
            height = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=230.0, value=float(profile["height"]), step=1.0)
        with col_w1:
            weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=250.0, value=float(profile["weight"]), step=0.5)

        activity_level = st.selectbox(
            "Tingkat Aktivitas Harian",
            options=list(ACTIVITY_MULTIPLIERS.keys()),
            index=2,
            help="Pilih intensitas aktivitas dan olahraga Anda dalam seminggu untuk akurasi TDEE.",
        )

        goal = st.selectbox(
            "Tujuan Kebugaran (Fitness Goal)",
            options=[
                "Pembentukan Otot (Hypertrophy / Bulking)",
                "Penurunan Lemak (Fat Loss / Cutting)",
                "Peningkatan Kekuatan (Strength)",
                "Kebugaran & Kesehatan Umum (Maintenance)",
            ],
            index=0 if "Otot" in profile["goal"] else 1,
            help="Target kebugaran Anda: Membesarkan otot (bulking), membakar lemak (cutting/fat loss), kekuatan (strength), atau menjaga kebugaran tubuh (maintenance).",
        )

        experience = st.selectbox(
            "Pengalaman Latihan",
            options=["Pemula (Beginner)", "Menengah (Intermediate)", "Mahir (Advanced)"],
            index=0,
            help="Tingkat jam terbang dan pengalaman Anda dalam latihan beban.",
        )

        location = st.selectbox(
            "Lokasi & Alat Latihan",
            options=[
                "Gym Komersial (Alat Lengkap)",
                "Rumah - Hanya Dumbbell",
                "Rumah - Bodyweight / Tanpa Alat",
                "Outdoor / Calisthenics Park",
            ],
            index=0,
            help="Fasilitas atau peralatan yang Anda miliki untuk menjalankan program latihan.",
        )

        if st.button("💾 Simpan & Update Profil", type="primary", use_container_width=True):
            st.session_state.profile = {
                "profile_complete": True,
                "gender": gender,
                "age": age,
                "height": height,
                "weight": weight,
                "activity_level": activity_level,
                "goal": goal,
                "experience": experience,
                "location": location,
            }
            st.toast("✅ Profil fisik berhasil disimpan!", icon="💪")
            st.rerun()

    # Hitung metrik biometrik secara real-time
    curr_p = st.session_state.profile
    bmi = calculate_bmi(curr_p["weight"], curr_p["height"])
    bmi_category, bmi_color, bmi_note = get_bmi_category(bmi)
    bmr = calculate_bmr(curr_p["gender"], curr_p["weight"], curr_p["height"], curr_p["age"])
    tdee = calculate_tdee(bmr, curr_p["activity_level"])
    nutrition = calculate_target_nutrition(tdee, curr_p["weight"], curr_p["goal"])

    # Tampilkan kartu metrik biometrik di sidebar
    st.markdown("#### 📊 Ringkasan Biometrik")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Indeks Massa Tubuh</div>
                <div class="metric-val" style="color: {bmi_color};">{bmi}</div>
                <div class="metric-sub" style="color: {bmi_color};">{bmi_category.split('(')[0]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Target Kalori</div>
                <div class="metric-val" style="color: #00E5FF;">{nutrition['target_calories']}</div>
                <div class="metric-sub" style="color: #9CA3AF;">kcal / hari</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="background:#111A2E; border-radius:8px; padding:10px; margin-top:8px; font-size:0.8rem; border-left:3px solid {bmi_color};">
            <b>Saran Biometrik:</b> {bmi_note}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Aksi Tambahan: Reset & Export
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_btn2:
        # Ekspor riwayat obrolan ke format Markdown
        chat_export = f"# FITCOACH AI - Riwayat Latihan\n\n"
        chat_export += generate_client_context_summary(st.session_state.profile) + "\n\n"
        chat_export += "## Riwayat Percakapan\n\n"
        for m in st.session_state.messages:
            sender = "Klien" if m["role"] == "user" else f"Coach ({selected_persona})"
            chat_export += f"### {sender}:\n{m['content']}\n\n"

        st.download_button(
            label="📥 Download .md",
            data=chat_export,
            file_name="fitcoach_program_latihan.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ==========================================
# MAIN CONTAINER: HERO BANNER
# ==========================================
coach_info = COACH_PERSONAS[selected_persona]

if curr_p["profile_complete"]:
    status_badge_html = f"""
    <div style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; background: rgba(0, 230, 118, 0.12); border: 1px solid rgba(0, 230, 118, 0.4); border-radius: 8px; color: #00E676; font-size: 0.85rem; font-weight: 600; margin-top: 10px;">
        <span>✅ Profil Aktif:</span> {curr_p['gender']}, {curr_p['age']} th, {curr_p['height']} cm, {curr_p['weight']} kg &bull; Target: {curr_p['goal']}
    </div>
    """
else:
    status_badge_html = """
    <div style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 8px; color: #FBBF24; font-size: 0.85rem; font-weight: 600; margin-top: 10px;">
        <span>⚠️ Profil Belum Lengkap:</span> Silakan isi jenis kelamin, usia, TB, dan BB di panel samping atau langsung ketik di chat.
    </div>
    """

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-badge">{coach_info['icon']} {coach_info['name']} Aktif</div>
        <h1 class="hero-title">FitCoach <span>AI</span></h1>
        <p class="hero-subtitle">
            Asisten gym personal bertenaga <b>Google Gemini</b>. Membantu menghitung komposisi tubuh,
            menjawab pertanyaan biomekanika, dan menyusun program latihan yang disesuaikan secara ilmiah.
        </p>
        {status_badge_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# QUICK ACTION BUTTONS
# ==========================================
st.markdown("##### ⚡ Pertanyaan Cepat & Template Program:")
col_q1, col_q2, col_q3, col_q4 = st.columns(4)

quick_prompt = None
with col_q1:
    if st.button("🏋️ Program Push-Pull-Legs 4 Hari", use_container_width=True):
        quick_prompt = "Tolong buatkan program latihan Push-Pull-Legs 4 hari seminggu yang disesuaikan dengan data fisik dan target kebugaran saya."
with col_q2:
    if st.button("🥗 Hitung Makro & Menu Makanan", use_container_width=True):
        quick_prompt = "Berapa gram protein, karbohidrat, dan lemak yang harus saya konsumsi setiap hari sesuai berat badan dan target saya? Berikan juga contoh menu makannya."
with col_q3:
    if st.button("🏠 Program Home Workout (No Alat)", use_container_width=True):
        quick_prompt = "Tolong buatkan program latihan efektif di rumah tanpa alat (bodyweight) selama 3-4 hari per minggu."
with col_q4:
    if st.button("⏱️ Pemanasan & Mobilitas Sendi", use_container_width=True):
        quick_prompt = "Bagaimana urutan pemanasan dinamis (dynamic warmup) dan mobilitas sendi yang benar sebelum mulai angkat beban agar tidak cedera?"


# ==========================================
# PESAN AWAL KETIKA CHAT MASIH KOSONG
# ==========================================
if not st.session_state.messages:
    if not curr_p["profile_complete"]:
        initial_greeting = f"""Halo! Saya **{coach_info['name']}**, pelatih kebugaran personal Anda di FitCoach AI. 🏋️‍♂️

Untuk memastikan program latihan aman dan efektif sesuai kondisi tubuh Anda, **mohon berikan 4 data fisik berikut terlebih dahulu**:
1. 🚻 **Jenis Kelamin** (Pria / Wanita)
2. 🎂 **Usia** saat ini
3. 📏 **Tinggi Badan** (cm)
4. ⚖️ **Berat Badan** (kg)

*(Anda juga bisa langsung mengisi formulir di panel samping sebelah kiri!)*"""
    else:
        initial_greeting = f"""Halo! Saya **{coach_info['name']}**, personal trainer Anda. 💪

Saya melihat profil Anda: **{curr_p['gender']}, {curr_p['age']} tahun, TB {curr_p['height']} cm, BB {curr_p['weight']} kg** dengan BMI **{bmi} ({bmi_category.split('(')[0]})**.
Kebutuhan kalori harian Anda berada di kisaran **{nutrition['target_calories']} kcal** dengan target protein harian **{nutrition['protein_g']} gram**.

Ada yang ingin kita latih atau susun hari ini? Silakan tanyakan program latihan, teknik gerakan, atau pola makan Anda!"""

    st.session_state.messages.append({"role": "model", "content": initial_greeting})


# ==========================================
# MENAMPILKAN RIWAYAT CHAT
# ==========================================
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar=coach_info["icon"]):
            st.markdown(msg["content"])


# ==========================================
# PROSES INPUT USER
# ==========================================
user_query = st.chat_input("Tanya program latihan, menu makan, atau ketik data tubuh (cth: Pria, 25 th, 175 cm, 70 kg)...")

# Jika tombol aksi cepat ditekan, gunakan teksnya sebagai query
if quick_prompt:
    user_query = quick_prompt

if user_query:
    # 1. Parsing otomatis jika user menyebutkan data fisik dalam chat
    updated = try_parse_biometrics_from_text(user_query)
    if updated:
        st.toast("💡 Data profil fisik Anda berhasil diperbarui dari chat!", icon="✅")

    # 2. Tambahkan pesan user ke histori
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    # 3. Bangun konteks dan instruksi sistem
    client_summary = generate_client_context_summary(st.session_state.profile)
    system_instruction = build_system_instruction(
        persona_key=selected_persona,
        client_summary=client_summary,
        profile_complete=st.session_state.profile["profile_complete"],
    )

    # 4. Stream respon dari Gemini API
    with st.chat_message("assistant", avatar=coach_info["icon"]):
        response_generator = stream_gemini_response(
            api_key=st.session_state.api_key,
            model_name=selected_model,
            system_instruction=system_instruction,
            messages=st.session_state.messages,
            temperature=temperature,
        )
        full_response = st.write_stream(response_generator)

    # 5. Simpan respon asisten ke riwayat chat
    if full_response:
        st.session_state.messages.append({"role": "model", "content": full_response})
