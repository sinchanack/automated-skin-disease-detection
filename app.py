import streamlit as st
import os
import uuid
import re
import pandas as pd
from PIL import Image
import keras
from keras import metrics
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64


# Register the custom metric function as serializable
@keras.saving.register_keras_serializable()
def top_2_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)


@keras.saving.register_keras_serializable()
def top_3_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


# Paths for model files
# Updated paths for model files
MODEL_JSON_PATH = r'F:\Skin-LesionDetection-main\Skin\Skin\model.json'
MODEL_WEIGHTS_PATH = r'F:\Skin-LesionDetection-main\Skin\Skin\model.h5'
HOME_BG_PATH = r'F:\Skin-LesionDetection-main\Skin\Skin\skinbg.jpg'
# Path for the Excel file to store user data
USER_DATA_PATH = r'F:\Skin-LesionDetection-main\Skin\Skin\user_data.xlsx'

# Load the model
try:
    with open(MODEL_JSON_PATH, 'r') as j_file:
        loaded_json_model = j_file.read()
    model = model_from_json(loaded_json_model, custom_objects={
                            'top_2_accuracy': top_2_accuracy, 'top_3_accuracy': top_3_accuracy})
    model.load_weights(MODEL_WEIGHTS_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Prediction classes
classes = [
    'Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis',
    'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular naevus',
]


# Utility functions
def is_valid_email(email):
    return re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email)


def is_strong_password(password):
    return len(password) >= 8 and re.search(r"[A-Z]", password) and re.search(r"[0-9]", password) and re.search(r"[@$!%*?&#]", password)


def predict_image(image_path, model, threshold=0.5):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32') / 255.0
    result = model.predict(img)

    predictions = {result[0][i]: classes[i] for i in range(len(result[0]))}
    sorted_results = sorted(predictions.items(),
                            key=lambda x: x[0], reverse=True)

    top_prob = sorted_results[0][0]
    top_class = sorted_results[0][1]

    if top_prob < threshold:
        top_classes = [item[1] for item in sorted_results[:3]]

    top_classes = [item[1] for item in sorted_results[:3]]
    top_probs = [(item[0] * 100).round(2) for item in sorted_results[:3]]

    return top_classes, top_probs, top_class, top_prob


# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'user' not in st.session_state:
    st.session_state.user = None


# Function to check if user data exists
def load_user_data():
    if os.path.exists(USER_DATA_PATH):
        return pd.read_excel(USER_DATA_PATH)
    else:
        return pd.DataFrame(columns=['email', 'name', 'password'])


# Function to save user data
def save_user_data(df):
    df.to_excel(USER_DATA_PATH, index=False)


# Function to navigate to a different page
def navigate_to(page):
    st.session_state.page = page
    st.rerun()


# Set background image using CSS
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded_image}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Sign-up Page
def sign_up_page():
    st.title("Sign Up")
    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    user_data = load_user_data()

    if st.button("Sign Up"):
        if not name.isalpha():
            st.error("Name should contain only alphabetic characters.")
        elif not is_valid_email(email):
            st.error("Invalid email format.")
        elif not is_strong_password(password):
            st.error("Password is not strong enough.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        elif email in user_data['email'].values:
            st.error("Email is already registered.")
        else:
            new_user = pd.DataFrame([[email, name, password]], columns=[
                                    'email', 'name', 'password'])
            user_data = pd.concat([user_data, new_user], ignore_index=True)
            save_user_data(user_data)
            st.success("Account created successfully!")
            navigate_to("Log In")

    # Navigation buttons at the bottom
    st.write("Already have an account?")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Log In"):
            navigate_to("Log In")
    with col3:
        if st.button("Home"):
            navigate_to("Home")


# Log-in Page
def log_in_page():
    st.title("Log In")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    user_data = load_user_data()

    if st.button("Log In"):
        if email in user_data['email'].values and user_data.loc[user_data['email'] == email, 'password'].values[0] == password:
            st.session_state.user = user_data.loc[user_data['email']
                                                  == email, 'name'].values[0]
            navigate_to("Home")
        else:
            st.error("Invalid email or password.")

    # Navigation buttons at the bottom
    st.write("Don't have an account?")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Sign Up"):
            navigate_to("Sign Up")
    with col3:
        if st.button("Home"):
            navigate_to("Home")


# Home Page
def home_page():
    set_background(HOME_BG_PATH)

    # Header for navigation with Sign In/Log In
    st.sidebar.title("Navigation")
    if st.session_state.user is None:
        if st.sidebar.button("Log In"):
            navigate_to("Log In")
        if st.sidebar.button("Sign Up"):
            navigate_to("Sign Up")
    else:
        st.sidebar.title(f"Welcome, {st.session_state.user}")
        if st.sidebar.button("Log Out"):
            st.session_state.user = None
            navigate_to("Home")

    # Main content of the home page
    st.title("Skin Disease Detection System")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Check Disease"):
            if st.session_state.user is None:
                st.warning("You need to sign up or log in to check diseases.")
            else:
                navigate_to("Check Disease")
    with col2:
        if st.button("Know About Diseases"):
            navigate_to("Know About Diseases")


# Check Disease Page
def check_disease_page():
    """Render the skin disease classification page."""
    st.title("Skin Disease Detection System")

    if st.button("Home"):
        navigate_to("Home")

    uploaded_file = st.file_uploader("Choose an image file", type=[
                                     'jpg', 'jpeg', 'png', 'jfif'])
    if uploaded_file:
        # Save the uploaded file temporarily
        unique_filename = str(uuid.uuid4()) + ".jpg"
        temp_file_path = os.path.join("temp", unique_filename)
        os.makedirs("temp", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Use columns for layout
        col1, col2 = st.columns(2)

        # Display the uploaded image on the left side
        with col1:
            st.write("### Uploaded Image")
            image = Image.open(temp_file_path)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict button and results on the right side
        with col2:
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    top_classes, top_probs, top_class, top_prob = predict_image(
                        temp_file_path, model)

                # Display results
                st.success("Prediction Completed!")

                # Split the space for predictions and final classification
                pred_col1, pred_col2 = st.columns([2, 1])
                with pred_col1:
                    st.write("### Top Predictions")
                    for i in range(len(top_classes)):
                        st.write(f"*{top_classes[i]}*: {top_probs[i]}%")
                with pred_col2:
                    st.write("### Final Classification")
                    st.write(
                        f"*Predicted Class*: {top_class} with {top_prob:.2f}%")

        # Clean up temporary file
        os.remove(temp_file_path)


# Know About Diseases Page
# Know About Diseases Page
def know_about_diseases_page():
    """Render the 'Know About Diseases' page."""
    st.title("Know About Diseases")

    # Navigation button back to Home
    if st.button("Home"):
        navigate_to("Home")


# Know About Diseases Page
# Know About Diseases Page
def know_about_diseases_page():
    """Render the 'Know About Diseases' page."""
    st.title("Know About Diseases")

    # Navigation button back to Home
    if st.button("Home"):
        navigate_to("Home")

    # Disease information dictionary
    diseases_info = {
        "Actinic Keratoses": {
            "description": (
                "Actinic Keratoses are rough, scaly patches on the skin that develop due to prolonged exposure to ultraviolet (UV) radiation from the sun or artificial sources like tanning beds. "
                "These lesions are considered precancerous, as they can occasionally progress to squamous cell carcinoma, a type of skin cancer. Commonly found on sun-exposed areas like the face, scalp, hands, and arms, they may appear red, pink, or brown and feel like sandpaper. "
                "Early detection and treatment, such as cryotherapy or topical medications, can prevent their progression."
            ),
            "symptoms": [
                "Rough, scaly patches on sun-exposed areas like the face, scalp, hands, and arms.",
                "May feel like sandpaper or appear red, brown, or pink.",
                "Sometimes itchy or tender."
            ],
            "causes": [
                "Prolonged sun exposure causing damage to skin cells.",
                "More common in fair-skinned individuals or those with a history of frequent sunburns."
            ],
            "solutions": [
                "Use sunscreen daily and wear protective clothing.",
                "Avoid excessive sun exposure, especially during peak hours.",
                "Topical creams or cryotherapy (freezing the spots) may be prescribed.",
                "Consult a doctor if patches grow, change color, or bleed."
            ]
        },
        "Basal Cell Carcinoma": {
            "description": (
                "Basal Cell Carcinoma is the most common and least aggressive form of skin cancer. It arises in the basal cells, which are located in the deepest layer of the epidermis. "
                "Caused primarily by prolonged UV exposure, BCC manifests as pearly or waxy bumps, flat lesions, or sores that may bleed and fail to heal. While BCC rarely spreads to other parts of the body, it can cause significant damage to surrounding tissues if left untreated. "
                "Early intervention through surgical removal, radiation therapy, or topical treatments is essential for effective management."
            ),
            "symptoms": [
                "Pearly or waxy bumps, often with visible blood vessels.",
                "Flat, flesh-colored, or brownish scar-like lesions.",
                "Sores that bleed, crust, and do not heal."
            ],
            "causes": [
                "UV radiation from sunlight or tanning beds.",
                "Chronic exposure to arsenic or radiation."
            ],
            "solutions": [
                "Surgical removal of the affected area.",
                "Treatments like topical creams, radiation, or photodynamic therapy.",
                "Regular skin check-ups.",
                "Consult a doctor immediately."
            ]
        },
        "Benign Keratosis": {
            "description": (
                "Benign Keratosis, also known as seborrheic keratosis, is a non-cancerous skin growth that often appears as a thickened, wart-like lesion. "
                "These growths are usually brown, black, or light tan and have a slightly raised, waxy texture. They tend to develop in older adults and may occur on the face, chest, shoulders, or back. "
                "While their exact cause is unclear, they may have a genetic component. Benign keratoses are harmless and typically do not require treatment, though removal for cosmetic reasons is possible."
            ),
            "symptoms": [
                "Thickened, wart-like patches that are usually brown, black, or light tan.",
                "Typically painless and found on the trunk, face, or shoulders."
            ],
            "causes": [
                "Age-related changes in skin cells.",
                "Possible genetic predisposition."
            ],
            "solutions": [
                "Usually harmless and may not need treatment.",
                "Removal options include cryotherapy or minor surgery for cosmetic purposes.",
                "Consult a doctor if the lesion changes in appearance or becomes bothersome."
            ]
        },
        "Dermatofibroma": {
            "description": (
                "Dermatofibroma is a small, firm, benign skin nodule that commonly appears on the legs or arms. These nodules are typically red, brown, or purple and may feel like a hard lump beneath the skin. "
                "Dermatofibromas are believed to form as a reaction to minor skin injuries, such as insect bites or trauma. They are generally painless but can be tender or itchy in some cases. "
                "As they pose no significant health risk, treatment is usually unnecessary, though surgical removal can be an option for persistent discomfort or cosmetic reasons."
            ),
            "symptoms": [
                "Firm, small, round nodules on the skin, often on the legs or arms.",
                "Can be red, brown, or purple and feel like a hard lump.",
                "Sometimes itchy or tender when touched."
            ],
            "causes": [
                "Exact cause unknown but may develop after minor skin injuries like insect bites or cuts.",
                "Common in adults, especially women."
            ],
            "solutions": [
                "Usually harmless and does not require treatment.",
                "Can be surgically removed if bothersome.",
                "Consult a doctor if you notice rapid growth, color change, or pain."
            ]
        },
        "Melanoma": {
            "description": (
                "Melanoma is a serious and potentially life-threatening form of skin cancer that develops in melanocytes, the cells responsible for producing pigment in the skin. "
                "It often begins as a mole that changes in size, shape, or color, displaying irregular borders or multiple hues. Commonly caused by intense UV exposure and genetic predisposition, melanoma can spread to other parts of the body if not detected early. "
                "Prompt diagnosis and treatment, including surgical removal and possibly immunotherapy or chemotherapy for advanced cases, are critical for favorable outcomes."
            ),
            "symptoms": [
                "New or changing moles with irregular shapes, multiple colors, or asymmetry.",
                "Lesions larger than 6mm in diameter or ones that bleed, itch, or scab."
            ],
            "causes": [
                "UV radiation from the sun or tanning beds.",
                "Genetic factors or a history of severe sunburns."
            ],
            "solutions": [
                "Early detection is key—surgical removal in early stages.",
                "Advanced cases may require chemotherapy, immunotherapy, or radiation.",
                "Use sunscreen and perform regular self-checks for unusual moles.",
                "Consult a doctor immediately, as melanoma is the most dangerous type of skin cancer."
            ]
        },
        "Melanocytic Nevi": {
            "description": (
                "Melanocytic Nevi, commonly known as moles, are benign growths formed by clusters of melanocytes, the pigment-producing cells in the skin. "
                "They can appear anywhere on the body and vary in color from light brown to black. Most moles are harmless and develop during childhood or adolescence, though some may change over time. "
                "Monitoring for irregularities in size, color, or shape is important, as these changes could signal melanoma. Cosmetic removal or monitoring by a dermatologist may be recommended in certain cases."
            ),
            "symptoms": [
                "Small, pigmented spots or growths on the skin, varying from light brown to black.",
                "Can be flat or raised and are generally round or oval."
            ],
            "causes": [
                "Accumulation of pigment-producing cells (melanocytes).",
                "May be genetic or influenced by sun exposure."
            ],
            "solutions": [
                "Regular monitoring to ensure no changes in size, shape, or color.",
                "Cosmetic removal if desired.",
                "Consult a doctor if a mole changes in appearance or begins to bleed, as it could indicate melanoma."
            ]
        },
        "Vascular naevi": {
            "description": (
                "Vascular naevi, also known as vascular birthmarks, are abnormalities of the blood vessels in the skin that result in red, pink, or purple marks at birth or shortly thereafter. These marks, such as port-wine stains or hemangiomas, are caused by clusters of blood vessels that grow abnormally. While many vascular birthmarks fade naturally over time, some persist and may require treatment for cosmetic or medical reasons. Laser therapy is a common option for managing prominent marks, especially those that grow, bleed, or affect daily life.",
            ),
            "symptoms": [
                "Red, pink, or purple patches or spots, commonly known as “strawberry marks” or “port-wine stains.”",
                "Can appear anywhere on the body, often noticeable at birth."
            ],
            "causes": [
                "Abnormal growth or clustering of blood vessels in the skin."
            ],
            "solutions": [
                "Many fade naturally with time, especially in children.",
                "Laser therapy for persistent or cosmetically concerning marks.",
                "Consult a Doctor: If the mark grows rapidly or bleeds frequently."
            ]
        },
    }
    st.sidebar.title("Choose a Disease")
    selected_disease = st.sidebar.selectbox(
        "Select a disease to learn about", diseases_info.keys())

    # Display selected disease information
    if selected_disease:
        st.header(selected_disease)
        disease = diseases_info[selected_disease]

        st.write("### Description")
        st.write(disease["description"])

        st.write("### Symptoms")
        for symptom in disease["symptoms"]:
            st.write(f"- {symptom}")

        st.write("### Causes")
        for cause in disease["causes"]:
            st.write(f"- {cause}")

        st.write("### Solutions")
        for solution in disease["solutions"]:
            st.write(f"- {solution}")


# Render the appropriate page based on the session state
if st.session_state.page == "Sign Up":
    sign_up_page()
elif st.session_state.page == "Log In":
    log_in_page()
elif st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Check Disease":
    check_disease_page()
elif st.session_state.page == "Know About Diseases":
    know_about_diseases_page()
