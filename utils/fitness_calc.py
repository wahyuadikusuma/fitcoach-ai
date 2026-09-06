"""
Modul Kalkulator Kebugaran (Fitness Calculator)
Menghitung BMI, BMR (Mifflin-St Jeor), TDEE, Kebutuhan Kalori Harian, dan Estimasi Makronutrien.
"""

from typing import Dict, Any, Tuple


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Menghitung Body Mass Index (BMI). Rumus: BB (kg) / (TB (m))^2"""
    if height_cm <= 0 or weight_kg <= 0:
        return 0.0
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 2)


def get_bmi_category(bmi: float) -> Tuple[str, str, str]:
    """
    Mengembalikan kategori BMI berdasarkan standar WHO / Kemenkes RI.
    Returns:
        tuple: (Kategori, Kode Warna Hex, Catatan Singkat)
    """
    if bmi <= 0:
        return ("Tidak Valid", "#9E9E9E", "Mohon masukkan data tinggi dan berat badan yang valid.")
    elif bmi < 18.5:
        return (
            "Kurus (Underweight)",
            "#00E5FF",
            "Perlu peningkatan asupan kalori bernutrisi dan latihan beban untuk menambah massa otot."
        )
    elif 18.5 <= bmi < 24.9:
        return (
            "Normal / Ideal",
            "#00E676",
            "Berat badan ideal! Fokus pada komposisi tubuh, kebugaran kardiovaskular, dan pembentukan otot."
        )
    elif 25.0 <= bmi < 29.9:
        return (
            "Kelebihan Berat Badan (Overweight)",
            "#FFD600",
            "Direkomendasikan defisit kalori moderat dipadu latihan beban & kardio teratur."
        )
    else:
        return (
            "Obesitas (Obese)",
            "#FF5252",
            "Perlu program penurunan lemak bertahap, pola makan terkontrol, dan aktivitas fisik terstruktur yang ramah persendian."
        )


def calculate_bmr(gender: str, weight_kg: float, height_cm: float, age: int) -> float:
    """
    Menghitung Basal Metabolic Rate (BMR) menggunakan persamaan Mifflin-St Jeor.
    Pria: 10 * BB + 6.25 * TB - 5 * Usia + 5
    Wanita: 10 * BB + 6.25 * TB - 5 * Usia - 161
    """
    if weight_kg <= 0 or height_cm <= 0 or age <= 0:
        return 0.0
    
    base = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age)
    if gender.strip().lower() in ["pria", "laki-laki", "male", "l"]:
        bmr = base + 5
    else:
        bmr = base - 161
    
    return round(bmr, 1)


ACTIVITY_MULTIPLIERS = {
    "Sedentary (Jarang / Tidak Pernah Olahraga)": 1.2,
    "Lightly Active (Olahraga Ringan 1-3 hari/minggu)": 1.375,
    "Moderately Active (Olahraga Sedang 3-5 hari/minggu)": 1.55,
    "Very Active (Olahraga Berat 6-7 hari/minggu)": 1.725,
    "Extremely Active (Latihan Atletik / Pekerjaan Fisik Berat)": 1.9,
}


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Menghitung Total Daily Energy Expenditure (TDEE)."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.375)
    return round(bmr * multiplier, 1)


def calculate_target_nutrition(tdee: float, weight_kg: float, goal: str) -> Dict[str, Any]:
    """
    Menghitung rekomendasi kalori harian dan pembagian makronutrien (Protein, Lemak, Karbohidrat).
    """
    goal_lower = goal.lower()
    
    if "fat loss" in goal_lower or "turun" in goal_lower:
        # Defisit kalori ~20%
        target_calories = round(tdee * 0.80)
        protein_ratio_per_kg = 2.0  # Menjaga massa otot saat defisit
        fat_pct = 0.25
        goal_label = "Penurunan Lemak (Defisit ~20%)"
    elif "muscle" in goal_lower or "hypertrophy" in goal_lower or "otot" in goal_lower or "bulking" in goal_lower:
        # Surplus kalori ~10-15%
        target_calories = round(tdee * 1.12)
        protein_ratio_per_kg = 1.8  # Pembentukan jaringan otot baru
        fat_pct = 0.25
        goal_label = "Pembentukan Otot / Lean Bulking (Surplus ~12%)"
    elif "strength" in goal_lower or "kekuatan" in goal_lower:
        target_calories = round(tdee * 1.05)
        protein_ratio_per_kg = 1.8
        fat_pct = 0.25
        goal_label = "Peningkatan Kekuatan / Strength"
    else:
        # Maintenance
        target_calories = round(tdee)
        protein_ratio_per_kg = 1.6
        fat_pct = 0.25
        goal_label = "Pemeliharaan & Kesehatan Umum (Maintenance)"

    # Hitung gram makro:
    # Protein: 4 kcal/g
    protein_g = round(weight_kg * protein_ratio_per_kg)
    protein_kcal = protein_g * 4

    # Lemak: 9 kcal/g (25% dari total kalori)
    fat_kcal = round(target_calories * fat_pct)
    fat_g = round(fat_kcal / 9)

    # Karbohidrat: sisa kalori (4 kcal/g)
    carbs_kcal = max(0, target_calories - protein_kcal - fat_kcal)
    carbs_g = round(carbs_kcal / 4)

    return {
        "goal_label": goal_label,
        "target_calories": target_calories,
        "protein_g": protein_g,
        "fat_g": fat_g,
        "carbs_g": carbs_g,
        "protein_kcal": protein_kcal,
        "fat_kcal": fat_kcal,
        "carbs_kcal": carbs_kcal,
    }


def generate_client_context_summary(profile: Dict[str, Any]) -> str:
    """Membuat ringkasan profil klien terstruktur untuk diinjeksikan ke system instruction LLM."""
    if not profile or not profile.get("profile_complete", False):
        return "Profil klien belum lengkap. Tanyakan jenis kelamin, usia, tinggi badan, dan berat badan kepada klien secara ramah."

    gender = profile.get("gender", "Tidak disebutkan")
    age = profile.get("age", 0)
    height = profile.get("height", 0.0)
    weight = profile.get("weight", 0.0)
    activity = profile.get("activity_level", "Moderately Active")
    goal = profile.get("goal", "General Fitness")
    experience = profile.get("experience", "Beginner")
    location = profile.get("location", "Gym Komersial")

    bmi = calculate_bmi(weight, height)
    bmi_cat, _, _ = get_bmi_category(bmi)
    bmr = calculate_bmr(gender, weight, height, age)
    tdee = calculate_tdee(bmr, activity)
    nutrition = calculate_target_nutrition(tdee, weight, goal)

    summary = f"""
=== DATA PROFIL KLIEN (VERIFIED BIOMETRICS) ===
- Jenis Kelamin: {gender}
- Usia: {age} tahun
- Tinggi Badan: {height} cm
- Berat Badan: {weight} kg
- BMI: {bmi} ({bmi_cat})
- BMR (Mifflin-St Jeor): {bmr} kcal/hari
- TDEE (Pengeluaran Energi Harian): {tdee} kcal/hari
- Target Utama: {goal} ({nutrition['goal_label']})
- Tingkat Pengalaman: {experience}
- Lokasi & Fasilitas Latihan: {location}
- Rekomendasi Target Kalori: {nutrition['target_calories']} kcal/hari
- Estimasi Makronutrien:
  * Protein: {nutrition['protein_g']}g ({nutrition['protein_kcal']} kcal)
  * Karbohidrat: {nutrition['carbs_g']}g ({nutrition['carbs_kcal']} kcal)
  * Lemak: {nutrition['fat_g']}g ({nutrition['fat_kcal']} kcal)
================================================
"""
    return summary.strip()
