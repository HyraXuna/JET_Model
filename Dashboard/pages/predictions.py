import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO

# ---------------------- FONCTIONS ----------------------

def preprocess_zone(cell, size=(224, 224)):
    cell = cv2.resize(cell, size, interpolation=cv2.INTER_AREA)
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
    array = img_to_array(cell)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)

def predict_wear(cells, model, seuil=0.6):
    labels, confidences = [], []
    for cell in cells:
        input_img = preprocess_zone(cell)
        pred = model.predict(input_img, verbose=0)
        if pred.shape[1] == 1:
            cls = int(pred[0][0] > seuil)
            conf = pred[0][0] if cls == 1 else 1 - pred[0][0]
        else:
            cls = np.argmax(pred)
            conf = pred[0][cls]
        labels.append(cls)
        confidences.append(conf)
    return labels, confidences

def grid_split_img(image, rows=4, cols=4):
    h, w, _ = image.shape
    cell_h, cell_w = h // rows, w // cols
    margin = 10
    cells = []
    for i in range(rows):
        for j in range(cols):
            y1, x1 = i * cell_h, j * cell_w
            y2, x2 = y1 + cell_h, x1 + cell_w
            y1, y2 = max(y1 - margin, 0), min(y2 + margin, h)
            x1, x2 = max(x1 - margin, 0), min(x2 + margin, w)
            cell = image[y1:y2, x1:x2]
            cells.append(cell)
    return cells

def show_prediction_grid(zones, labels, confidences, class_names):
    st.markdown("<div class='centered-grid'>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, cell in enumerate(zones):
        label = class_names[labels[i]]
        conf = confidences[i]
        color = "🟩" if label == "bon" else "🟥"
        with cols[i % 4]:
            st.image(Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)), width=160)
            st.markdown(f"<div style='text-align:center'><b>Zone {i+1}</b><br>{color} {label} ({conf:.2f})</div>", unsafe_allow_html=True)

def generate_heatmap_overlay(base_image, confidences, rows=4, cols=4, alpha=0.4):
    matrix = np.array(confidences).reshape((rows, cols))
    h, w = base_image.shape[:2]
    heatmap_resized = cv2.resize(matrix, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_norm = np.clip(heatmap_resized, 0, 1)
    heatmap_colored = cv2.applyColorMap((255 * (1 - heatmap_norm)).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay, heatmap_norm

# ---------------------- INTERFACE ----------------------


st.markdown("""
    <style>
    .centered-grid { max-width: 1200px; margin: 0 auto; }
    div[data-testid="column"] { padding: 0.5rem !important; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Analyse de l'usure des pneus")

uploaded_image = st.file_uploader("📷 Téléversez une image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    img_col, result_col = st.columns([0.9, 2.1]) 

    with img_col:
        st.image(image, caption="Image originale", use_container_width=False)

    with st.spinner("Chargement du modèle..."):
        model_path = tf.keras.utils.get_file(
            "mobilenetv2_finetune.h5",
            origin="https://huggingface.co/HyraXuna/Jet_model_MobileNetV2/resolve/main/model_epoch_23_val_acc_0.86.h5",
            cache_subdir='models'
        )
        model = tf.keras.models.load_model(model_path)
        yolo_path = hf_hub_download(repo_id="flodussart/jet_yolov8m", filename="best.pt")
        yolo_model = YOLO(yolo_path)

    resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(resized) / 255.0, axis=0)
    global_pred = model.predict(img_array)[0][0]
    global_conf = global_pred if global_pred > 0.5 else 1 - global_pred

    with result_col:
        st.subheader("Prédiction globale")
        if global_pred < 0.5:
            st.error(f"❌ Le pneu semble **usé** ({global_conf:.2f})")
        else:
            st.success(f"✅ Le pneu semble **bon** ({global_conf:.2f})")

        st.divider()

        results = yolo_model(image_cv2)[0]
        if len(results.boxes) > 0:
            box = results.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image_cv2.shape[1]), min(y2, image_cv2.shape[0])

            zoom = image_cv2[y1:y2, x1:x2]
            zones = grid_split_img(zoom)
            labels, confidences = predict_wear(zones, model)
            class_names = ["usé", "bon"]

            st.subheader("🧩 Prédiction par zones")
            show_prediction_grid(zones, labels, confidences, class_names)

        
            # 🌡️ Carte thermique
            # st.subheader("🌡️ Carte thermique de l'usure")

            # # Génère overlay et heatmap (valeurs entre 0 et 1)
            # overlay, heatmap_matrix = generate_heatmap_overlay(zoom, confidences)

            # # Affichage avec matplotlib
            # fig, ax = plt.subplots(figsize=(6, 6))

            # # 1. Image superposée en fond
            # ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            # ax.axis("off")

            # # 2. Image invisible pour barre (on la rend visible par cmap et alpha=1 dans colorbar seulement)
            # im = ax.imshow(heatmap_matrix, cmap='viridis', alpha=0)

            # # 3. Barre d'échelle directement liée à cette image
            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # cbar.set_label("Usure (0 = usé, 1 = bon)", rotation=270, labelpad=15)
            # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

            # # Affichage Streamlit
            # st.pyplot(fig)




        else:
            st.warning("Aucun pneu détecté.")
