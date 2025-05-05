import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# Dictionary of remedies (same as your file, shortened here for brevity)
disease_remedies ={
    'Apple___Apple_scab': """
    **Remedy:**
    1. Remove and destroy infected leaves and fruit
    2. Apply fungicides containing sulfur or copper
    3. Prune trees to improve air circulation
    4. Water at the base of the plant, not on leaves
    5. Use resistant apple varieties

    **Additional Solutions:**
    - Apply fungicides in early spring before bud break
    - Use organic alternatives like neem oil or baking soda solution
    - Maintain proper tree nutrition with balanced fertilization
    - Monitor weather conditions and apply preventive treatments before wet periods
    - Consider using biological control agents like Bacillus subtilis

    **Preventive Measures:**
    - Plant trees in well-drained soil with good air circulation
    - Space trees properly to reduce humidity
    - Remove fallen leaves and fruit in autumn
    - Apply dormant oil spray in late winter
    - Monitor trees regularly for early signs of infection
   
    **Recommended Pesticides:**
    - Captan
    - Myclobutanil
    - Mancozeb
    - Sulfur (organic)
    - Copper-based sprays
    """,
    'Apple___Black_rot': """
    **Remedy:**
    1. Remove and destroy infected fruit and branches
    2. Apply fungicides during bloom and fruit development
    3. Prune trees to improve air circulation
    4. Remove fallen leaves and fruit from the ground
    5. Use resistant apple varieties

    **Additional Solutions:**
    - Apply copper-based fungicides in early spring
    - Use biological control agents like Trichoderma
    - Implement proper irrigation practices
    - Monitor fruit development closely
    - Apply calcium sprays to strengthen fruit

    **Preventive Measures:**
    - Prune trees annually to maintain good structure
    - Remove mummified fruit from trees
    - Clean pruning tools between cuts
    - Maintain proper tree nutrition
    - Monitor weather conditions for infection risk
   
    **Recommended Pesticides:**
    - Ziram
    - Captan
    - Thiophanate-methyl
    - Copper fungicide
    - Trifloxystrobin
    """,
    'Apple___Cedar_apple_rust': """
    **Remedy:**
    1. Remove nearby cedar trees if possible
    2. Apply fungicides in early spring
    3. Use resistant apple varieties
    4. Prune infected branches
    5. Maintain good tree health through proper fertilization

    **Additional Solutions:**
    - Apply fungicides before and after rain events
    - Use organic alternatives like sulfur or copper
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Consider using biological control agents

    **Preventive Measures:**
    - Plant resistant apple varieties
    - Maintain proper tree spacing
    - Remove nearby juniper/cedar trees
    - Apply preventive fungicides in early spring
    - Monitor weather conditions for infection risk
   
    **Recommended Pesticides:**
    - Myclobutanil
    - Mancozeb
    - Sulfur
    - Copper oxychloride
    - Propiconazole
    """,
    'Apple___healthy': """
    **Maintenance Tips:**
    1. Continue regular monitoring for early signs of disease
    2. Maintain proper pruning schedule
    3. Implement balanced fertilization program
    4. Monitor soil moisture and irrigation needs
    5. Keep records of tree health and growth

    **Preventive Care:**
    - Apply dormant oil spray in late winter
    - Monitor for pest activity
    - Maintain proper tree spacing
    - Implement proper irrigation practices
    - Keep the area around trees clean and weed-free
   
    **Recommended Pesticides:**
    - No routine pesticides needed; monitor and apply only if symptoms appear
    """,
    'Blueberry___healthy': """
    **Maintenance Tips:**
    1. Monitor soil pH and maintain between 4.5-5.5
    2. Implement proper pruning schedule
    3. Maintain consistent moisture levels
    4. Apply appropriate fertilizers
    5. Monitor for pest activity

    **Preventive Care:**
    - Apply mulch to maintain soil moisture
    - Monitor for signs of nutrient deficiencies
    - Implement proper irrigation practices
    - Keep the area around plants clean
    - Monitor for early signs of disease
   
    **Recommended Pesticides:**
    - No routine pesticides needed; monitor for pests and fungal signs
    """,
    'Cherry_(including_sour)___Powdery_mildew': """
    **Remedy:**
    1. Apply sulfur-based fungicides
    2. Prune to improve air circulation
    3. Remove infected leaves
    4. Water at the base of the plant
    5. Use resistant varieties

    **Additional Solutions:**
    - Apply horticultural oil sprays
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply preventive treatments before infection

    **Preventive Measures:**
    - Plant resistant varieties
    - Maintain proper tree spacing
    - Prune annually for good air circulation
    - Monitor weather conditions
    - Implement proper fertilization program
   
    **Recommended Pesticides:**
    - Sulfur
    - Trifloxystrobin
    - Myclobutanil
    - Neem oil
    - Potassium bicarbonate
    """,
    'Cherry_(including_sour)___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper pruning schedule
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of tree health

    **Preventive Care:**
    - Apply dormant oil spray in late winter
    - Monitor for pest activity
    - Maintain proper tree spacing
    - Implement proper irrigation practices
    - Keep the area around trees clean
   
    **Recommended Pesticides:**
    - No routine pesticides needed; use dormant oil in winter
    """,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': """
    **Remedy:**
    1. Use resistant corn varieties
    2. Rotate crops
    3. Apply fungicides when necessary
    4. Remove crop debris after harvest
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive fungicides before symptoms appear
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Practice crop rotation with non-host crops
    - Use disease-free seeds
    - Maintain proper plant nutrition
    - Monitor weather conditions
    - Implement proper field sanitation
   
    **Recommended Pesticides:**
    - Pyraclostrobin
    - Azoxystrobin
    - Propiconazole
    - Trifloxystrobin
    - Mancozeb
    """,
    'Corn_(maize)___Common_rust_': """
    **Remedy:**
    1. Use resistant corn varieties
    2. Apply fungicides if necessary
    3. Remove infected plants
    4. Practice crop rotation
    5. Maintain proper plant nutrition

    **Additional Solutions:**
    - Apply preventive fungicides before symptoms appear
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Plant resistant varieties
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
   
    **Recommended Pesticides:**
    - Tebuconazole
    - Azoxystrobin
    - Propiconazole
    - Mancozeb
    - Chlorothalonil
    """,
    'Corn_(maize)___Northern_Leaf_Blight': """
    **Remedy:**
    1. Use resistant corn varieties
    2. Practice crop rotation
    3. Remove and destroy infected plant debris
    4. Apply fungicides if necessary
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive fungicides before symptoms appear
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Plant resistant varieties
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
   
    **Recommended Pesticides:**
    - Azoxystrobin
    - Pyraclostrobin
    - Propiconazole
    - Trifloxystrobin
    - Mancozeb
    """,
    'Corn_(maize)___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper plant spacing
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of plant health

    **Preventive Care:**
    - Practice crop rotation
    - Use disease-free seeds
    - Maintain proper plant nutrition
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Grape___Black_rot': """
    **Remedy:**
    1. Apply fungicides during bloom and fruit development
    2. Remove and destroy infected fruit and leaves
    3. Prune to improve air circulation
    4. Use resistant grape varieties
    5. Remove fallen leaves and fruit from the ground

    **Additional Solutions:**
    - Apply preventive fungicides before symptoms appear
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Plant resistant varieties
    - Maintain proper vine spacing
    - Prune annually for good air circulation
    - Monitor weather conditions
    - Implement proper vineyard sanitation
    """,
    'Grape___Esca_(Black_Measles)': """
    **Remedy:**
    1. Prune infected vines
    2. Apply fungicides to pruning wounds
    3. Remove and destroy infected plants
    4. Use disease-free planting material
    5. Maintain proper vine nutrition

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free planting material
    - Maintain proper vine spacing
    - Prune carefully to avoid large wounds
    - Monitor weather conditions
    - Implement proper vineyard sanitation
    """,
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': """
    **Remedy:**
    1. Apply fungicides during wet weather
    2. Remove infected leaves
    3. Prune to improve air circulation
    4. Use resistant grape varieties
    5. Maintain proper vine nutrition

    **Additional Solutions:**
    - Apply preventive fungicides before symptoms appear
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Plant resistant varieties
    - Maintain proper vine spacing
    - Prune annually for good air circulation
    - Monitor weather conditions
    - Implement proper vineyard sanitation
    """,
    'Grape___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper pruning schedule
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of vine health

    **Preventive Care:**
    - Apply dormant spray in late winter
    - Monitor for pest activity
    - Maintain proper vine spacing
    - Implement proper irrigation practices
    - Keep the vineyard clean and weed-free
    """,
    'Orange___Haunglongbing_(Citrus_greening)': """
    **Remedy:**
    1. Remove and destroy infected trees
    2. Control psyllid vectors with insecticides
    3. Use disease-free planting material
    4. Monitor for symptoms regularly
    5. Maintain proper tree nutrition

    **Additional Solutions:**
    - Apply systemic insecticides
    - Use biological control agents for psyllids
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free planting material
    - Maintain proper tree spacing
    - Monitor for psyllid activity
    - Implement proper orchard sanitation
    - Use reflective mulches to deter psyllids
    """,
    'Peach___Bacterial_spot': """
    **Remedy:**
    1. Apply copper-based bactericides
    2. Prune to improve air circulation
    3. Remove infected leaves and fruit
    4. Use resistant peach varieties
    5. Maintain proper tree nutrition

    **Additional Solutions:**
    - Apply preventive bactericides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Plant resistant varieties
    - Maintain proper tree spacing
    - Prune annually for good air circulation
    - Monitor weather conditions
    - Implement proper orchard sanitation
    """,
    'Peach___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper pruning schedule
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of tree health

    **Preventive Care:**
    - Apply dormant spray in late winter
    - Monitor for pest activity
    - Maintain proper tree spacing
    - Implement proper irrigation practices
    - Keep the orchard clean and weed-free
    """,
    'Pepper,_bell___Bacterial_spot': """
    **Remedy:**
    1. Use disease-free seeds and transplants
    2. Apply copper-based bactericides
    3. Remove infected plants
    4. Practice crop rotation
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive bactericides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Pepper,_bell___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper plant spacing
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of plant health

    **Preventive Care:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant nutrition
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Potato___Early_blight': """
    **Remedy:**
    1. Apply fungicides containing chlorothalonil or mancozeb
    2. Remove infected leaves
    3. Practice crop rotation
    4. Use resistant potato varieties
    5. Maintain proper plant nutrition

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seed potatoes
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Potato___Late_blight': """
    **Remedy:**
    1. Apply fungicides containing chlorothalonil or mancozeb
    2. Remove and destroy infected plants
    3. Practice crop rotation
    4. Use resistant potato varieties
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seed potatoes
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Potato___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper plant spacing
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of plant health

    **Preventive Care:**
    - Use disease-free seed potatoes
    - Practice crop rotation
    - Maintain proper plant nutrition
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Raspberry___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper pruning schedule
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of plant health

    **Preventive Care:**
    - Apply dormant spray in late winter
    - Monitor for pest activity
    - Maintain proper plant spacing
    - Implement proper irrigation practices
    - Keep the area around plants clean
    """,
    'Soybean___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper plant spacing
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of plant health

    **Preventive Care:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant nutrition
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Squash___Powdery_mildew': """
    **Remedy:**
    1. Apply sulfur-based fungicides
    2. Remove infected leaves
    3. Improve air circulation
    4. Water at the base of the plant
    5. Use resistant varieties

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Plant resistant varieties
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    - Use reflective mulches
    """,
    'Strawberry___Leaf_scorch': """
    **Remedy:**
    1. Apply fungicides containing chlorothalonil
    2. Remove infected leaves
    3. Improve air circulation
    4. Use resistant strawberry varieties
    5. Maintain proper plant nutrition

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Plant resistant varieties
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    - Use clean planting material
    """,
    'Strawberry___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper plant spacing
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of plant health

    **Preventive Care:**
    - Use disease-free plants
    - Practice crop rotation
    - Maintain proper plant nutrition
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Tomato___Bacterial_spot': """
    **Remedy:**
    1. Use disease-free seeds and transplants
    2. Apply copper-based bactericides
    3. Remove infected plants
    4. Practice crop rotation
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive bactericides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Tomato___Early_blight': """
    **Remedy:**
    1. Apply fungicides containing chlorothalonil
    2. Remove infected leaves
    3. Practice crop rotation
    4. Use resistant tomato varieties
    5. Maintain proper plant nutrition

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Tomato___Late_blight': """
    **Remedy:**
    1. Apply fungicides containing chlorothalonil
    2. Remove and destroy infected plants
    3. Practice crop rotation
    4. Use resistant tomato varieties
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Tomato___Leaf_Mold': """
    **Remedy:**
    1. Improve air circulation
    2. Apply fungicides containing chlorothalonil
    3. Remove infected leaves
    4. Use resistant tomato varieties
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Tomato___Septoria_leaf_spot': """
    **Remedy:**
    1. Apply fungicides containing chlorothalonil
    2. Remove infected leaves
    3. Practice crop rotation
    4. Use resistant tomato varieties
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Tomato___Spider_mites Two-spotted_spider_mite': """
    **Remedy:**
    1. Apply insecticidal soap or neem oil
    2. Introduce predatory mites
    3. Increase humidity
    4. Remove heavily infested leaves
    5. Maintain proper plant nutrition

    **Additional Solutions:**
    - Apply horticultural oils
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Monitor for early signs of infestation
    - Maintain proper plant spacing
    - Keep plants well-watered
    - Use reflective mulches
    - Implement proper field sanitation
    """,
    'Tomato___Target_Spot': """
    **Remedy:**
    1. Apply fungicides containing chlorothalonil
    2. Remove infected leaves
    3. Practice crop rotation
    4. Use resistant tomato varieties
    5. Maintain proper plant spacing

    **Additional Solutions:**
    - Apply preventive fungicides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor weather conditions
    - Implement proper field sanitation
    """,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': """
    **Remedy:**
    1. Control whitefly populations
    2. Remove infected plants
    3. Use resistant tomato varieties
    4. Use reflective mulches
    5. Maintain proper plant nutrition

    **Additional Solutions:**
    - Apply systemic insecticides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor for whitefly activity
    - Implement proper field sanitation
    """,
    'Tomato___Tomato_mosaic_virus': """
    **Remedy:**
    1. Remove infected plants
    2. Control aphid populations
    3. Use disease-free seeds
    4. Practice good sanitation
    5. Use resistant tomato varieties

    **Additional Solutions:**
    - Apply systemic insecticides
    - Use biological control agents
    - Implement proper irrigation practices
    - Monitor for early symptoms
    - Apply balanced fertilization

    **Preventive Measures:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant spacing
    - Monitor for aphid activity
    - Implement proper field sanitation
    """,
    'Tomato___healthy': """
    **Maintenance Tips:**
    1. Monitor for signs of disease and pests
    2. Maintain proper plant spacing
    3. Implement balanced fertilization
    4. Monitor soil moisture
    5. Keep records of plant health

    **Preventive Care:**
    - Use disease-free seeds
    - Practice crop rotation
    - Maintain proper plant nutrition
    - Monitor weather conditions
    - Implement proper field sanitation
    """
}

# List of class names (as you already had)
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Upload an image or capture using ESP32-CAM to detect plant diseases and get remedies.
    
    ### Features
    - 📷 Upload images or capture live via ESP32-CAM
    - 🚀 Instant prediction
    - 🛠️ Remedies and preventive tips
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    - 87,000+ plant images
    - 38 disease categories
    - Trained with Tensorflow
    - Data from original PlantVillage Dataset + Augmentation

    #### Project Goal
    Early plant disease detection using deep learning models with user-friendly interfaces.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("🌱 Disease Recognition")

    option = st.radio("Choose input method:", ("Upload Image", "ESP32-CAM Capture"))

    if option == "Upload Image":
        test_image = st.file_uploader("Choose an Image:")

        if test_image is not None:
            if st.button("Show Image"):
                st.image(test_image, use_container_width=True)

            if st.button("Predict"):
                with st.spinner("Analyzing Image..."):
                    result_index = model_prediction(test_image)
                    predicted_disease = class_name[result_index]

                    st.success(f"✅ Prediction: **{predicted_disease}**")
                    st.markdown("### Recommended Remedy")
                    st.markdown(disease_remedies[predicted_disease])

    elif option == "ESP32-CAM Capture":
        esp32_url = st.text_input("Enter ESP32-CAM Snapshot URL:", "http://192.168.208.231/cam-hi.jpg")

        if st.button("📷 Capture & Predict"):
            try:
                response = requests.get(esp32_url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert('RGB')

                st.image(img, caption="Captured Image", use_column_width=True)
                st.info("Analyzing Captured Image...")

                # Save temporarily
                temp_path = "temp.jpg"
                img.save(temp_path)

                result_index = model_prediction(temp_path)
                predicted_disease = class_name[result_index]

                st.success(f"✅ Prediction: **{predicted_disease}**")
                st.markdown("### Recommended Remedy")
                st.markdown(disease_remedies[predicted_disease])

            except Exception as e:
                st.error(f"🚫 Error capturing from ESP32-CAM: {e}")
