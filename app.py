import streamlit as st
import joblib
import pandas as pd


# Modeli yükle
actor_xgb_model = joblib.load('actor_xgb1_model.pkl')
actress_lr_model = joblib.load('actress_lr_model.pkl')
director_xgb_model = joblib.load('director_xgb1_model.pkl')

# Başlık ekle
st.title("Yapay Zeka ile Show Dünyası Annus Mirabilis Tahmini")
st.markdown("""
Show dünyasında **"annus mirabilis"** terimi, bir aktörün veya sanatçının kariyerinde en başarılı ve 
en üretken yılı ifade eder. Latince kökenli olan bu terim, kelime anlamı olarak "mucize yıl" veya 
"harika yıl" anlamına gelir. 
"Annus mirabilis", sanatçının kariyerinde özel bir yere sahiptir ve 
genellikle bir sanatçının yeteneklerini, azmini ve şansını birleştirdiği dönem olarak hatırlanır. 
Bu yıl, sanatçının kariyerindeki diğer yıllardan daha belirgin ve hatırlanabilir olup, 
genellikle kariyerinin geri kalanını da olumlu yönde etkiler.""")
st.markdown("""
## Bilgilendirme
- **n (Total Movies)**: Kariyeri boyunca toplam yaptığı iş sayısı
- **s (Active Years)**: Aktif olarak iş yaptığı yıl sayısı. Aynı zamanda s = L — τ olarak bulabiliriz.
- **τ (Latent years)**: Kariyerinde iş yapmadığı senelerin toplam sayısı.
- **L (Career Length)**: İlk yaptığı iş ile son yaptığı iş arasındaki geçen yıl sayısı.
- **Am=m (Max Value)**: Kariyerinde bir senede yaptığı en yüksek iş sayısı. Yani oyuncunun en iyi yılı.
- **startYear**: Oyuncunun kariyerine başladığı ilk yıl.
""")
# Kullanıcıdan veri girişi almak için giriş alanları oluştur
def get_user_input():
    feature0 = st.number_input("Total Movies", min_value=0, max_value=1000, value=0)
    feature1 = st.number_input("Active Years", min_value=0, max_value=1000, value=0)
    feature2 = st.number_input("Latent Years", min_value=0, max_value=1000, value=0)
    feature3 = st.number_input("Career Length", min_value=0, max_value=1000, value=0)
    feature4 = st.number_input("Max Value", min_value=0, max_value=1000, value=0)
    feature5 = st.number_input("startYear", min_value=0, max_value=2024, value=0)
    # Diğer özellikleri buraya ekleyebilirsiniz
    data = {'Total Movies': feature0,
            'Active Years': feature1,
            'Latent Years': feature2,
            'Career Length': feature3,
            'Max Value': feature4,
            'startYear':feature5}
    features = pd.DataFrame(data, index=[0])
    return features

# Model seçimi için seçenek ekle
model_choice = st.selectbox(
    "Tahmin modeli seçin:",
    ("Aktör", "Aktris", "Yönetmen")
)

# Kullanıcıdan alınan verileri dataframe'e çevir
input_df = get_user_input()

# Kullanıcı verilerini göster
st.subheader("Kullanıcı Girdileri:")
st.write(input_df)

# Tahmin yap
if st.button("Tahmin Yap"):
    if model_choice == "Aktör":
        prediction = actor_xgb_model.predict(input_df)
    elif model_choice == "Aktris":
        prediction = actress_lr_model.predict(input_df)
    elif model_choice == "Yönetmen":
        prediction = director_xgb_model.predict(input_df)

    # Tahmin sonucunu değerlendirerek uygun metni yazdır
    st.subheader("Tahmin Sonucu:")
    if prediction == 1:
        st.write("Oyuncu en iyi yılında.")
    elif prediction == 0:
        st.write("Oyuncu henüz en iyi yılına ulaşamadı.")