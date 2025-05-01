import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model


st.title('EyeScan Pro : Votre Diagnostic Visuel en Un Clic')
# 1. Charger le modèle
try:
    modele = load_model("milleur_model_vgg16_adam.h5")
    st.sidebar.success("✅ Modèle chargé avec succès")
except Exception as e:
    st.sidebar.error(f"❌ Erreur de chargement du modèle : {e}")
    st.stop()

# 2. Classes prédictibles
classes = ['Diabetic Retinopathy', 'Glaucoma', 'Healthy', 'Macular Scar', 'Myopia']

# 3. Préparation de l'image
def preparer_image(img):
    try:
        img = img.resize((224, 224)).convert("RGB")  # Redimensionner à 224x224
        img_array = np.array(img) / 255.0  # Normalisation
        img_array = np.expand_dims(img_array, axis=0)  # Forme (1, 224, 224, 3)
        
        # Affichage des dimensions avant et après le prétraitement
        st.write(f"Image avant préparation : {img.size}")
        st.write(f"Forme de l'image après préparation : {img_array.shape}")
        
        return img_array
    except Exception as e:
        st.error(f"Erreur de prétraitement : {e}")
        return None

# 4. Fonction pour la prédiction d'image
def page_predire_image():
    # Vérification si les informations d'inscription sont remplies
    if "nom" not in st.session_state or "prenom" not in st.session_state:
        st.warning("⚠️ Vous devez d'abord remplir le formulaire d'inscription.")
        return

    st.title("🖼️ Prédiction des maladies des yeux")

    st.subheader("📷 Charger une image d'œil")

    fichier_image = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

    if fichier_image is not None:
        try:
            image = Image.open(fichier_image).convert("RGB")
            st.image(image, caption="Image chargée", use_column_width=True)

            img_prep = preparer_image(image)
            if img_prep is None:
                return

            # Prédiction
            try:
                prediction = modele.predict(img_prep)
                st.write(f"🧪 Valeurs de la prédiction : {prediction}")
                st.write(f"Forme de la prédiction : {prediction.shape}")
                
                # Vérifier que la sortie du modèle correspond aux classes
                if prediction.shape[1] != len(classes):
                    st.error(f"❌ La sortie du modèle ({prediction.shape[1]}) ne correspond pas au nombre de classes ({len(classes)}).")
                else:
                    classe_predite = int(np.argmax(prediction))
                    proba = float(np.max(prediction)) * 100
                    st.markdown(
                        f"<h3 style='color:green;'>✅ Résultat : {classes[classe_predite]} ({proba:.2f}%)</h3>",
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {e}")
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de l'image : {e}")

# 5. Fonction pour l'inscription
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
        st.success(f"Bienvenue {prenom} {nom} 👋")
        
        # Sauvegarder les informations de l'utilisateur dans la session
        st.session_state.nom = nom
        st.session_state.prenom = prenom
        st.session_state.adresse = adresse
        st.session_state.email = email
        st.session_state.genre = genre

        st.info("✅ Formulaire d'inscription soumis avec succès. Vous pouvez maintenant accéder à la page de prédiction.")

# 6. Menu de navigation pour les pages
PAGES = {
    "Page d'Inscription": page_inscription,
    "Prédiction d'Image": page_predire_image
}

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Sélectionner une page", list(PAGES.keys()))
    PAGES[page]()

if __name__ == "__main__":
    main()
