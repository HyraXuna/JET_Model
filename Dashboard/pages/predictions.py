

from PIL import Image
import streamlit as st
import random
import tensorflow as tf
import numpy as np
import json
import os

page_icon = Image.open("logo/logo_app_favicon.png")


# Logo + titre côte à côte
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo/logo_app_favicon.png", width=50)
with col2:
    st.markdown("## Choisissez une image")

# Upload d’image
uploaded_image = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])


if uploaded_image is not None:
    # st.success("Image chargée avec succès !")
    
    # Ouvrir et redimensionner l'image
    image = Image.open(uploaded_image)
    max_width = 224
    w_percent = (max_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    resized_image = image.resize((max_width, h_size))

    # Préparer l'image pour la prédiction
    img_array = np.array(resized_image)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch
    img_for_pred = img_array / 255.0  # Normaliser l'image

    # # Afficher l'image redimensionnée
    # st.image(resized_image, caption="Aperçu de l'image", use_container_width=False)

# Layout en colonnes : image à gauche, résultat à droite
    img_col, result_col = st.columns([1.2, 1])

    with img_col:
        st.image(resized_image, caption="Aperçu de l'image", use_container_width=False)

    # Modèle
        def model_prediction(img):
            model_path = tf.keras.utils.get_file(
                "mobilenetv2_finetune.h5",
                origin="https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/model_epoch_23_val_acc_0.86.h5",
                cache_subdir='models'
            )

            if not os.path.exists(model_path):
                st.error("Le modèle n'a pas pu être téléchargé.")
                return None, None

            model_loaded = tf.keras.models.load_model(model_path)

            
            history_path = tf.keras.utils.get_file(
                "mibilenetv2_finetune_History.json",
                origin="https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/mobilenetv2model_finetune_History.json",
                cache_subdir='models'
            )

            if not os.path.exists(history_path):
                st.error("L'historique du modèle n'a pas pu être téléchargé.")
                return None, None

            with open(history_path, "r") as f:
                metrics = json.load(f)
            result = model_loaded.predict(img)
            result = (result > 0.5).astype(int).flatten()
            confidence = round(metrics["val_binary_accuracy"][22], 2)
            return result, confidence
        
    with result_col:
        if st.button("Analyser 🧠"):
            result, confidence = model_prediction(img_for_pred)
            #st.write(result)

            st.markdown("### Résultat de l’analyse :")

            if result == [0]:
                st.error(f"❌ Le pneu semble **usé**\n\n{confidence * 100:.0f}% de confiance")
            else:
                st.success(f"✅ Le pneu semble **en bon état**\n\n{confidence * 100:.0f}% de confiance")

    
