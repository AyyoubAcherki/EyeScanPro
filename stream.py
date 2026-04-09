import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
import gdown  # pip install gdown

# # === CONFIGURATION DU MODÈLE ===
# drive_file_id = "1Av6VaKPSGTbv0ed2-OsPwaGVr2aEEH6o"
# model_path = "milleur_model_vgg16_adam.h5"

# # === TÉLÉCHARGEMENT DU MODÈLE DEPUIS GOOGLE DRIVE (si besoin) ===
# if not os.path.exists(model_path):
#     st.sidebar.info("📥 Téléchargement du modèle depuis Google Drive...")
#     try:
#         url = f"https://drive.google.com/uc?id={drive_file_id}"
#         gdown.download(url, model_path, quiet=False)
#         st.sidebar.success("✅ Modèle téléchargé avec succès.")
#     except Exception as e:
#         st.sidebar.error(f"❌ Erreur de téléchargement du modèle : {e}")
#         st.stop()

# # === CHARGEMENT DU MODÈLE ===
# try:
#     modele = load_model(model_path)
#     st.sidebar.success("✅ Modèle chargé avec succès")
# except Exception as e:
#     st.sidebar.error(f"❌ Erreur de chargement du modèle : {e}")
#     st.stop()

# # === CLASSES POSSIBLES ===
# classes = ['Diabetic Retinopathy', 'Glaucoma', 'Healthy', 'Macular Scar', 'Myopia']

# # === PRÉPARATION DE L'IMAGE ===
# def preparer_image(img):
#     try:
#         img = img.resize((224, 224)).convert("RGB")
#         img_array = np.array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)
#         return img_array
#     except Exception as e:
#         st.error(f"Erreur de préparation de l'image : {e}")
#         return None

import streamlit as st
from tensorflow.keras.models import load_model
import os
import gdown
import numpy as np
from PIL import Image

# === CONFIGURATION DU MODÈLE ===
drive_file_id = "1Av6VaKPSGTbv0ed2-OsPwaGVr2aEEH6o"
model_path_h5 = "milleur_model_vgg16_adam.h5"
model_path_keras = "milleur_model_vgg16_adam.keras"  # conversion recommandée

# === TÉLÉCHARGEMENT DU MODÈLE DEPUIS GOOGLE DRIVE (si besoin) ===
if not os.path.exists(model_path_h5) and not os.path.exists(model_path_keras):
    st.sidebar.info("📥 Téléchargement du modèle depuis Google Drive...")
    try:
        url = f"https://drive.google.com/uc?id={drive_file_id}"
        gdown.download(url, model_path_h5, quiet=False)
        st.sidebar.success("✅ Modèle téléchargé avec succès.")
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de téléchargement du modèle : {e}")
        st.stop()

# === CHARGEMENT DU MODÈLE AVEC CACHE STREAMLIT ===
@st.cache_resource
def charger_modele():
    if os.path.exists(model_path_keras):
        return load_model(model_path_keras)
    else:
        # Si uniquement le .h5 existe, on charge avec compile=False pour éviter les erreurs
        return load_model(model_path_h5, compile=False)

try:
    modele = charger_modele()
    st.sidebar.success("✅ Modèle chargé avec succès")
except Exception as e:
    st.sidebar.error("❌ Erreur de chargement du modèle : vérifier la compatibilité TensorFlow/Keras")
    st.sidebar.error(str(e))
    st.stop()

# === CLASSES POSSIBLES ===
classes = ['Diabetic Retinopathy', 'Glaucoma', 'Healthy', 'Macular Scar', 'Myopia']

# === PRÉPARATION DE L'IMAGE ===
def preparer_image(img):
    try:
        img = img.resize((224, 224)).convert("RGB")
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Erreur de préparation de l'image : {e}")
        return None
        
# === PAGE : FORMULAIRE D'INSCRIPTION ===
def page_inscription():
    st.title("🧾 Formulaire d'inscription")

    with st.form("formulaire"):
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        adresse = st.text_input("Adresse")
        email = st.text_input("Email")
        genre = st.radio("Genre", ["Homme", "Femme"])
        soumis = st.form_submit_button("Valider")

    if soumis:
        if not all([nom, prenom, adresse, email]):
            st.warning("⚠️ Veuillez remplir tous les champs.")
            return

        # Stocker les infos dans la session
        st.session_state.nom = nom
        st.session_state.prenom = prenom
        st.session_state.adresse = adresse
        st.session_state.email = email
        st.session_state.genre = genre

        st.success(f"Bienvenue {prenom} {nom} 👋")
        st.info("✅ Formulaire validé. Accédez maintenant à la prédiction.")

# === PAGE : PRÉDICTION D'IMAGE ===
def page_predire_image():
    if "nom" not in st.session_state or "prenom" not in st.session_state:
        st.warning("⚠️ Merci de remplir d'abord le formulaire d'inscription.")
        return

    st.title("🔬 Prédiction des maladies oculaires")
    st.subheader("📷 Importer une image d'œil")

    fichier_image = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

    if fichier_image:
        try:
            image = Image.open(fichier_image).convert("RGB")
            st.image(image, caption="Image chargée", use_column_width=True)

            img_prep = preparer_image(image)
            if img_prep is None:
                return

            prediction = modele.predict(img_prep)
            classe_predite = np.argmax(prediction)
            proba = np.max(prediction) * 100
            maladie = classes[classe_predite]

            st.markdown(
                f"<h3 style='color:green;'>✅ Résultat : {maladie} ({proba:.2f}%)</h3>",
                unsafe_allow_html=True
            )

            # Affichage optionnel brut
            st.write("🧪 Détails bruts de la prédiction :", prediction)

        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")

# === MENU DE NAVIGATION ===
PAGES = {
    "Page d'inscription": page_inscription,
    "Prédiction d'image": page_predire_image
}

def main():
    st.sidebar.title("🧭 Navigation")
    choix = st.sidebar.radio("Choisir une page", list(PAGES.keys()))
    PAGES[choix]()

if __name__ == "__main__":
    main()
