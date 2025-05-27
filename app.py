import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Aminata_CNN.h5")
    return model

model = load_model()

class_names = ['blessé', 'non_blessé'] 


st.title("🩺 Détection d'animaux blessés")

st.write("Téléversez une image pour savoir si l'animal est blessé ou non.")


uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  

    prediction = model.predict(img_array)[0][0]

    seuil = 0.76
    classe_predite = class_names[1] if prediction > seuil else class_names[0]
    confiance = prediction if prediction > seuil else 1 - prediction

    st.markdown(f"### 🧠 Résultat : **{classe_predite.upper()}**")
    st.markdown(f"Confiance du modèle : `{confiance:.2f}`")

    if classe_predite == 'blessé':
        st.warning("⚠️ L'animal semble blessé.")
    else:
        st.success("✅ L'animal semble en bonne santé.")
