"""
Modul Integrasi Gemini API
Mendukung official google-genai SDK dan google-generativeai SDK dengan streaming response,
persona pelatih dinamis, dan context injection profil biometrik klien.
"""

import os
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

load_dotenv()

# Persona prompt configuration
COACH_PERSONAS = {
    "Motivator Energik (Coach Alex)": {
        "name": "Coach Alex",
        "style": "Sangat energik, antusias, penuh semangat dan dorongan mental. Sering menggunakan kata-kata pemberi semangat seperti 'Gaspol!', 'Luar biasa!', 'Jangan kasih kendor!'. Fokus pada konsistensi dan mindset juara.",
        "icon": "🔥",
    },
    "Drill Sergeant (Coach Viktor)": {
        "name": "Coach Viktor",
        "style": "Tegas, disiplin tinggi, militeristik, straight to the point, tanpa alasan ('No Excuses!'). Selalu menuntut komitmen penuh dan eksekusi form yang presisi.",
        "icon": "⚡",
    },
    "Sports Scientist (Dr. Bryan, CSCS)": {
        "name": "Dr. Bryan, CSCS",
        "style": "Ilmiah, analitis, berbasis riset terkini (evidence-based). Menjelaskan mekanisme hipertrofi, RPE (Rate of Perceived Exertion), biomekanika sendi, nutrisi makro/mikro, serta fisiologi pemulihan otot secara sistematis.",
        "icon": "🧬",
    },
    "Empathetic & Friendly (Coach Maya)": {
        "name": "Coach Maya",
        "style": "Sangat ramah, penuh empati, sabar, dan pengertian. Sangat cocok untuk pemula yang baru pertama kali ke gym. Mengutamakan kenyamanan, membangun kebiasaan positif secara bertahap, dan bebas intimidasi.",
        "icon": "🌱",
    },
}

AVAILABLE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]


def build_system_instruction(
    persona_key: str,
    client_summary: str,
    profile_complete: bool,
) -> str:
    """Membangun instruksi sistem (system prompt) yang terpersonalisasi."""
    persona = COACH_PERSONAS.get(persona_key, COACH_PERSONAS["Motivator Energik (Coach Alex)"])
    
    instruction = f"""
Kamu adalah {persona['name']}, seorang Personal Trainer Gym bersertifikat internasional (CSCS - Certified Strength and Conditioning Specialist) dan Ahli Nutrisi Kebugaran.

=== GAYA BAHASA & PERSONA ===
Karakter kamu: {persona['style']}
Bicaralah dalam Bahasa Indonesia yang komunikatif sesuai karakter persona di atas.

=== STATUS PROFIL KLIEN ===
{client_summary}

=== ATURAN UTAMA INTERAKSI (CRITICAL PROTOCOL) ===
1. JIKA PROFIL KLIEN BELUM LENGKAP:
   - Jika pengguna belum memberikan informasi mengenai Jenis Kelamin, Usia, Tinggi Badan, dan Berat Badan, kamu HARUS menanyakan 4 data penting ini terlebih dahulu dengan ramah sebelum memberikan program latihan spesifik.
   - Jelaskan bahwa 4 data fisik ini mutlak diperlukan untuk menghitung kalori harian, indeks massa tubuh (BMI), dan menyusun beban latihan yang aman bagi persendian mereka.
   - Jika pengguna hanya memberikan sebagian data, tanyakan sisa data yang belum ada.

2. JIKA PROFIL KLIEN SUDAH LENGKAP:
   - Manfaatkan data biometrik yang tertera (BMI, BMR, TDEE, Kebutuhan Kalori) dalam setiap saran latihan dan pola makan.
   - Saat memberikan program latihan, gunakan format yang rapi dan terstruktur:
     * Hari & Pembagian Otot (Split)
     * Nama Gerakan (Exercise)
     * Jumlah Set x Repetisi (misal: 3-4 set x 8-12 reps)
     * Istirahat Antar Set (Rest time)
     * Catatan Form & Teknik Keamanan
   - Tekankan pentingnya Pemanasan (Warm-up / Dynamic Stretching) sebelum latihan dan Pendinginan (Cool-down / Static Stretching) setelahnya.
   - Berikan tips nutrisi praktis yang sesuai dengan target kalori dan gram protein yang telah dihitung.

3. KESELAMATAN & MEDIS:
   - Selalu utamakan keselamatan teknik gerak (*form over weight*).
   - Berikan *disclaimer* medis ramah bahwa jika klien memiliki cedera masa lalu atau kondisi medis khusus, mereka disarankan berkonsultasi dengan dokter fisioterapi.
"""
    return instruction.strip()


def stream_gemini_response(
    api_key: str,
    model_name: str,
    system_instruction: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Menghasilkan respon streaming dari Gemini API.
    Mendukung google-genai SDK terbaru dan google-generativeai.
    """
    if not api_key:
        yield "⚠️ **API Key belum diatur.** Silakan masukkan Google Gemini API Key Anda di panel samping (sidebar) atau atur di file `.env`."
        return

    # Metode 1: Menggunakan google-genai (v1.0+)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        # Siapkan riwayat chat
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )

        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        )

        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
        return

    except ImportError:
        pass
    except Exception as e_genai:
        # Coba fallback ke google-generativeai jika google-genai menemui kendala
        last_error = str(e_genai)

    # Metode 2: Menggunakan google-generativeai (legacy)
    try:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=api_key)

        model = legacy_genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config={"temperature": temperature},
        )

        # Ubah format history untuk legacy SDK
        chat_history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=chat_history)
        latest_message = messages[-1]["content"] if messages else ""

        response = chat.send_message(latest_message, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
        return

    except Exception as e_legacy:
        yield f"❌ **Terjadi kesalahan saat menghubungi Gemini API:**\n\n`{str(e_legacy)}`\n\nPastikan API key valid dan memiliki kuota yang tersedia."
