import streamlit as st
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

# Load a pre-trained VGG-16 model

# load YAML and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

def classify_food(image_path, top_n=3):
    # Load an image for prediction
    img = Image.open(image_path)

    # Resize the image to the target size
    img = img.resize((384, 384))


    # Preprocess the image for the model
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get predictions from the model
    model= loaded_model
    predictions = model.predict(img_array)

# Reshape predictions to (samples, 1000)
    predictions = np.reshape(predictions, (1, 1000))

    # Decode the predictions using decode_predictions
    decoded_predictions = decode_predictions(predictions, top=top_n)[0]

    return decoded_predictions

# Streamlit UI
st.title("Food Classification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Perform classification
    predictions = classify_food(uploaded_file)

    st.write("Top predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        st.write(f"{i + 1}: {label} ({score:.2f})")
