import streamlit as st
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf

st.title("Présentation et informations sur la baseline, le 1er modèle utilisé, le plus simple mais correct 🤖")
           
history_path = tf.keras.utils.get_file(
    "inceptionV3modelHistory.json",
    origin="https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/resolve/main/inceptionV3modelHistory.json",
    cache_subdir='models'
    )

if not os.path.exists(history_path):
    st.error("L'historique du modèle n'a pas pu être téléchargé.")

with open(history_path, "r") as f:
    metrics = json.load(f)



st.markdown("""
            Pour commencer, le modèle initial est le plus simpliste qui nous a donné des résultats corrects. 
            C'est un modèle de deep learning utilisant du transfer learning, c'est à dire que l'architecture est basée sur un modèle préentrainé.
            Dans notre cas, il se base sur **InceptionV3**, préentrainé avec les images de *imagenet*.
""")

st.markdown("""
            Ce modèle est stocké à cette adresse : https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/tree/main

            Si vous désirez le télécharger, veuillez cliquez ici : [télécharger le modèle 🦊](https://huggingface.co/HyraXuna/JET_model_inceptionV3_base/resolve/main/inceptionV3model.h5)
""")

st.subheader("Schéma du modèle 🪪")

# Affichage du schéma de structure en HTML
st.markdown("""
<table style="width:100%; border-collapse: collapse;">
  <tr style="border: 1px solid white;">
    <th style="border: 1px solid white; padding: 8px; text-align: left;"><b>Layer (type)</b></th>
    <th style="border: 1px solid white; padding: 8px; text-align: left;"><b>Output Shape</b></th>
    <th style="border: 1px solid white; padding: 8px; text-align: left;"><b>Param #</b></th>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">inception_v3 (<span style="color: #0087ff;">Functional</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">5</span>, <span style="color: #00af00;">5</span>, <span style="color: #00af00;">2048</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">21,802,784</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">global_average_pooling2d_2 (<span style="color: #0087ff;">GlobalAveragePooling2D</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">2048</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">0</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">dense_6 (<span style="color: #0087ff;">Dense</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">256</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">524,544</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">dense_7 (<span style="color: #0087ff;">Dense</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">128</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">32,896</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">dropout_2 (<span style="color: #0087ff;">Dropout</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">128</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">0</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">dense_8 (<span style="color: #0087ff;">Dense</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">1</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">129</span></td>
  </tr>
</table>
            
Total params: <span style="color: #00af00;">22,360,353</span> (85.30 MB)
            

Trainable params: <span style="color: #00af00;">557,569</span> (2.13 MB)
            

Non-trainable params: <span style="color: #00af00;">21,802,784</span> (83.17 MB)
            
""", unsafe_allow_html=True)

st.markdown("""
            Le modèle a été entrainé avec l'optimizer `Adam` avec un learning rate de $1^{-5}$, 
            ainsi qu'un calcul de fonction de coût et des métriques pour une classification binaire 
            (`BinaryCrossentropy` & `BinaryAccuracy`).
""")


st.subheader("Performances du modèle 📈")

col1, col2 = st.columns(2)

with col1:

    # Visualise train / Valid Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["binary_accuracy"], c="r", label="train_accuracy")
    plt.plot(metrics["val_binary_accuracy"], c="b", label="test_accuracy")
    plt.title("Accuracy of the model, Train Vs Validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

     # Afficher le graphique dans Streamlit
    st.pyplot(plt)

with col2:

    # Visualise train / Valid Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["loss"], c="r", label="train_loss")
    plt.plot(metrics["val_loss"], c="b", label="test_loss")
    plt.title("Loss of the model, Train Vs Validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

     # Afficher le graphique dans Streamlit
    st.pyplot(plt)

st.write("")
st.markdown("""
            Et maintenant voici la matrice de confusion correspondant à ce modèle, avec les données de validations:
""")


col1, col2, col3 = st.columns([1, 2, 1])

# Utiliser la colonne centrale pour afficher l'image
with col2:
    st.image("pages/confusion_matrix/matrix_confusion_inceptionv3.png", use_container_width=True)


st.title("Présentation et informations sur le modèle final 🤖")

history_path2 = tf.keras.utils.get_file(
    "mobilenetv2_fine_tune_History.json",
    origin="https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/mobilenetv2model_finetune_History.json",
    cache_subdir='models'
    )

if not os.path.exists(history_path2):
    st.error("L'historique du modèle n'a pas pu être téléchargé.")

with open(history_path2, "r") as f:
    metrics2 = json.load(f)

st.markdown("""
            Ce modèle plus performant, donnant de meilleurs résultats, utilise aussi du transfer learning, mais cette fois-ci basé sur 
            le modèle **MobileNetV2** préentrainé avec les images de *imagenet*. 
            Du fine tuning a été effectué ici pour améliorer les performance et permettre un apprentissage plus poussé sur nos données.

            Le modèle a été sauvegardé à chaque epoch où son score progressait. Donc pour la prédiction, nous avons utilisé celui sauvegardé
            à l'epoch 23 avec un score de validation de 0.8622.
            """)

st.markdown("""
            Ce modèle est stocké à cette adresse : https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/tree/main

            Si vous désirez le télécharger le modèle entier, à la fin des epoch, veuillez cliquez ici : [télécharger le modèle à la dernière epoch 🦊](https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/mobilenetv2model_finetune.h5)

            i vous désirez le télécharger le modèle au meilleur checkpoint, à l'epoch 23, veuillez cliquez ici : [télécharger le modèle à la meilleure epoch 🐯](https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/model_epoch_23_val_acc_0.86.h5)
""")

st.subheader("Schéma du modèle 🪪")

# Affichage du schéma de structure en HTML
st.markdown("""
<table style="width:100%; border-collapse: collapse;">
  <tr style="border: 1px solid white;">
    <th style="border: 1px solid white; padding: 8px; text-align: left;"><b>Layer (type)</b></th>
    <th style="border: 1px solid white; padding: 8px; text-align: left;"><b>Output Shape</b></th>
    <th style="border: 1px solid white; padding: 8px; text-align: left;"><b>Param #</b></th>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">mobilenetv2_1.00_224 (<span style="color: #0087ff;">Functional</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">7</span>, <span style="color: #00af00;">7</span>, <span style="color: #00af00;">1280</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">2,257,984</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">global_average_pooling2d (<span style="color: #0087ff;">GlobalAveragePooling2D</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">1280</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">0</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">dense (<span style="color: #0087ff;">Dense</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">256</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">327,936</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">dense_1 (<span style="color: #0087ff;">Dense</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">128</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">32,896</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">dropout (<span style="color: #0087ff;">Dropout</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">128</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">0</span></td>
  </tr>
  <tr style="border: 1px solid white;">
    <td style="border: 1px solid white; padding: 8px;">dense_2 (<span style="color: #0087ff;">Dense</span>)</td>
    <td style="border: 1px solid white; padding: 8px;">(<span style="color: #00d7ff;">None</span>, <span style="color: #00af00;">1</span>)</td>
    <td style="border: 1px solid white; padding: 8px;"><span style="color: #00af00;">129</span></td>
  </tr>
</table>

<br/>

<b>Total params:</b> <span style="color: #00af00;">2,618,945</span> (9.99 MB)  
<br/>
<b>Trainable params:</b> <span style="color: #00af00;">1,093,441</span> (4.17 MB)  
<br/>
<b>Non-trainable params:</b> <span style="color: #00af00;">1,525,504</span> (5.82 MB)
""", unsafe_allow_html=True)



st.markdown("""
            Le modèle a été entrainé avec l'optimizer `Adam` avec un learning rate de $1^{-5}$, 
            ainsi qu'un calcul de fonction de coût et des métriques pour une classification binaire 
            (`BinaryCrossentropy` & `BinaryAccuracy`). Il y avait aussi un `early stopping` basé sur la fonction de coût de la validation
            qui arrêtait l'entrainement du modèle s'il n"y avait pas de progression sur 5 epoch.
            Pour le `Fine tuning` nous avons libérer les 10 dernières couches du MibileNetV2.
""")


st.subheader("Performances du modèle 📈")

col1, col2 = st.columns(2)

with col1:

    # Visualise train / Valid Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics2["binary_accuracy"], c="r", label="train_accuracy")
    plt.plot(metrics2["val_binary_accuracy"], c="b", label="test_accuracy")
    plt.title("Accuracy of the model, Train Vs Validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

     # Afficher le graphique dans Streamlit
    st.pyplot(plt)

with col2:

    # Visualise train / Valid loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics2["loss"], c="r", label="train_loss")
    plt.plot(metrics2["val_loss"], c="b", label="test_loss")
    plt.title("Loss of the model, Train Vs Validation")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

     # Afficher le graphique dans Streamlit
    st.pyplot(plt)

st.write("")
st.markdown("""
            Et maintenant voici la matrice de confusion correspondant à ce modèle, avec les données de validations:
""")

col1, col2, col3 = st.columns([1, 2, 1])

# Utiliser la colonne centrale pour afficher l'image
with col2:
    st.image("pages/confusion_matrix/matrix_confusion_mobilinetv2_finetune.png", use_container_width=True)
    st.write("On voit bien une amélioration dans la prédiction par rapport à la baseline ! ✅")

