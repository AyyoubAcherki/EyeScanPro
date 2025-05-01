import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
import gdown
import logging

# ⚠️ Configuration à mettre en tout premier
st.set_page_config(page_title="EyeScan Pro", page_icon="👁️", layout="wide")

# === AFFICHAGE FIXE : logo + titre ===
def afficher_entete():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("11.jpeg", width=150)  # Ton logo ici
    with col2:
        st.markdown("<h1 style='color:#2C3E50;'>EyeScan Pro</h1>", unsafe_allow_html=True)

# === 1. Chargement du modèle ===
def charger_modele():
    model_path = "models/meilleur_model_vgg16_adam.h5"
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(model_path):
        try:
            st.sidebar.warning("⚠️ Téléchargement du modèle...")
            url = "https://drive.google.com/uc?id=1MYgwEtP5tkGe-wLPqRFSS7cDBmKz5Vwi"
            gdown.download(url, model_path, quiet=False)
            st.sidebar.success("✅ Modèle téléchargé !")
        except Exception as e:
            st.sidebar.error(f"❌ Échec du téléchargement : {str(e)}")
            st.stop()
    
    try:
        model = load_model(model_path)
        st.sidebar.success("✅ Modèle chargé avec succès")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de chargement : {str(e)}")
        st.stop()

modele = charger_modele()

# === 2. Dictionnaire des classes ===
CLASSES = {
    0: {'name': 'Diabetic Retinopathy', 'color': 'red'},
    1: {'name': 'Glaucoma', 'color': 'orange'},
    2: {'name': 'Healthy', 'color': 'green'},
    3: {'name': 'Macular Scar', 'color': 'purple'},
    4: {'name': 'Myopia', 'color': 'blue'}
}

# === 3. Prétraitement image ===
def preparer_image(img):
    try:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size[0] < 50 or img.size[1] < 50:
            st.warning("⚠️ Image trop petite - qualité de prédiction réduite")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        if np.max(img_array) > 1.0 or np.min(img_array) < 0.0:
            st.error("Erreur de normalisation des pixels")
            return None
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Erreur de prétraitement : {str(e)}")
        return None

# === 4. Page d’analyse ===
def page_predire_image():
    afficher_entete()
    if not all(k in st.session_state for k in ['nom', 'prenom']):
        st.warning("ℹ️ Veuillez compléter le formulaire d'inscription d'abord")
        return

    st.subheader("Analyse d'une image rétinienne")

    with st.expander("📸 Upload d'image", expanded=True):
        fichier = st.file_uploader("Choisissez une image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if fichier:
        try:
            img = Image.open(fichier)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Image originale", use_column_width=True)
            with col2:
                with st.spinner("Analyse en cours..."):
                    img_prep = preparer_image(img)
                    if img_prep is not None:
                        prediction = modele.predict(img_prep)
                        classe_idx = np.argmax(prediction)
                        confidence = np.max(prediction) * 100
                        classe = CLASSES[classe_idx]
                        st.markdown(
                            f"<h2 style='color:{classe['color']};'>"
                            f"Résultat: \n{classe['name']} ({confidence:.1f}%)</h2>",
                            unsafe_allow_html=True
                        )
                        probas = {CLASSES[i]['name']: float(prediction[0][i]) for i in CLASSES}
                        st.bar_chart(probas)
        except Exception as e:
            st.error(f"Erreur d'analyse: {str(e)}")

# === 5. Page d’inscription ===
def page_inscription():
    afficher_entete()
    st.subheader("Formulaire d'inscription du patient")
    with st.form("inscription"):
        cols = st.columns(2)
        prenom = cols[0].text_input("Prénom*", key="prenom_input")
        nom = cols[1].text_input("Nom*", key="nom_input")
        email = st.text_input("Email*", type="default")
        age = st.number_input("Âge", min_value=0, max_value=120)
        genre = st.radio("Genre", ["Homme", "Femme"], horizontal=True)
        submitted = st.form_submit_button("Sauvegarder")
        if submitted:
            if not all([prenom, nom, email]):
                st.error("Les champs obligatoires (*) doivent être remplis")
            else:
                st.session_state.update({
                    'prenom': prenom,
                    'nom': nom,
                    'email': email,
                    'age': age,
                    'genre': genre
                })
                st.success("Profil enregistré avec succès !")
                st.balloons()

# === 6. Navigation principale ===
def main():
    st.sidebar.header("Navigation")
    pages = {
        "📝 Inscription": page_inscription,
        "🔍 Analyse": page_predire_image
    }

    if "prenom" not in st.session_state:
        st.sidebar.warning("Complétez d'abord l'inscription")
        page = "📝 Inscription"
    else:
        page = st.sidebar.radio("Pages", list(pages.keys()))
    
    pages[page]()

if __name__ == "__main__":
    main()
