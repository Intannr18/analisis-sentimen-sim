import streamlit as st
import joblib

# ================= CONFIG =================
st.set_page_config(
    page_title="Analisis Sentimen Pelayanan SIM",
    page_icon="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhaQsQ6aHTsmlE-xsayZampsB9BbRW7jfnnRlRIBtRBuzOdhLVV9mGl0hVet7UMlj7heJGUdz_-RoRaWWMiRbZv6hqlV3Tq_uS3F5qpUVh5CGRfaptF8-6twBggJPNz5ZQh9MqOtg95u0k/s1600/polisi-vector-idngrafis.png",
    layout="centered"
)

# ================= LOAD MODEL =================
model = joblib.load("model/svm_model_new.pkl")
tfidf = joblib.load("model/tfidf_new.pkl")

# ================= STYLE =================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
    background-color: #0e1117;
    color: white;
}
.stApp {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}
.main-content {
    flex: 1;
}
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #9aa4b2;
    margin-bottom: 45px;
}
.big-label {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 12px;
}
textarea {
    background-color: #161b22 !important;
    color: white !important;
    border-radius: 14px !important;
    font-size: 16px !important;
}
textarea:focus {
    outline: none !important;
    border: 2px solid #00c6ff !important;
    box-shadow: 0 0 10px rgba(0,198,255,0.6) !important;
}
.stButton {
    margin-top: 18px;
}
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 15px;
    font-weight: 600;
    padding: 8px 22px;
    border-radius: 30px;
    border: none;
    box-shadow: 0 0 15px rgba(0,198,255,0.7);
    transition: 0.3s ease-in-out;
}
.stButton > button:hover {
    box-shadow: 0 0 24px rgba(34,197,94,1);
    transform: scale(1.05);
}
.result-label {
    margin-top: 40px;
    font-size: 20px;
    color: #9aa4b2;
}
.result {
    font-size: 32px;
    font-weight: 700;
    margin-top: 6px;
}
.footer {
    text-align: center;
    padding: 15px 0;
    color: #6b7280;
    font-size: 13px;
    border-top: 1px solid #1f2933;
}
</style>
""", unsafe_allow_html=True)

# ================= UI =================
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

st.markdown("<div class='title'>Analisis Sentimen Pelayanan SIM</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Sistem ini dibuat untuk mengukur tingkat kepuasan masyarakat terhadap pelayanan SIM</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='big-label'>Masukkan teks komentar</div>", unsafe_allow_html=True)

input_text = st.text_area(
    "",
    height=120,
    placeholder="Contoh: Puas banget sama pelayanan SIM tadi"
)

# ================= PREDIKSI =================
if st.button("Prediksi Sentimen"):
    if input_text.strip() == "":
        st.warning("Warning: Teks tidak boleh kosong!")
    else:
        vector = tfidf.transform([input_text])
        pred = model.predict(vector)[0].lower()

        # ===== FORCE 2 KELAS (NETRAL → POSITIF) =====
        if pred == "neutral":
            pred = "positive"

        emoji = {
            "positive": " ",
            "negative": " "
        }

        label = pred.capitalize()

        st.markdown(
            "<div class='result-label'>Hasil Analisis bernilai :</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='result'>{emoji[pred]} {label}</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>© 2025 | Sistem Analisis Sentimen Layanan SIM | Kelompok 3</div>",
    unsafe_allow_html=True
)
