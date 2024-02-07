'''
This food classification project is a commendable effort, considering the constraints on computational power. Limiting the classes to six ("chicken_curry," "chicken_wings," "french_fries,"
"Ice_Cream," "Pizza," and "samosa") demonstrates a practical approach to model training and deployment. While a larger dataset with 101 classes could potentially enhance the model's accuracy and versatility,
the decision to prioritize efficiency and feasibility is understandable.

For those seeking to expand the model's capabilities, the option to train it on the broader dataset of 101 food items provides an avenue for further exploration. 
This modular approach allows for scalability based on available resources and project requirements. Overall, this project showcases adaptability and resourcefulness in the face of computational limitations,
while still striving for effective food classification.

Link to data set = link(https://www.kaggle.com/datasets/kmader/food41?resource=download)
'''




import streamlit as st
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

# Load a pre-trained VGG16 model

# load YAML and create model
json_file = open('E:/coding_raja/food_classification/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("E:/coding_raja/food_classification/model.h5")
print("Loaded model from disk")

def classify_food(image_path, top_n=3):
    try:
        # Load an image for prediction
        img = Image.open(image_path)

        # Resize the image to the target size (modify this based on your model's input size)
        target_size = (384, 384)
        img = img.resize(target_size)

        # Preprocess the image for the model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get predictions from the model
        model = loaded_model
        predictions = model.predict(img_array)

        # Get the top predicted class index
        top_class_index = np.argmax(predictions[0])

        # You should have a mapping of class indices to labels
        class_labels = ["chicken_curry", "chicken_wings", "french_fries", "Ice_Cream", "Pizza", "samosa"]

        # Get the corresponding label for the top class
        top_class_label = class_labels[top_class_index]

        # Return the top class label and its confidence score
        return [(top_class_index, top_class_label, predictions[0][top_class_index])]

    except Exception as e:
        print("Error during classification:", str(e))
        return []

# Streamlit UI
st.title("Food Classification App")

# Display model information
st.sidebar.header("Model Information")
st.sidebar.text("VGG16 Model for Food Classification")
print("inside streamlit")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        # Perform classification
        predictions = classify_food(uploaded_file)

        st.write("Top predictions:")
        
        # Display predictions in a table
        predictions_table = "<table><tr><th>Rank</th><th>Label</th><th>Score</th></tr>"
        for i, (imagenet_id, label, score) in enumerate(predictions):
            predictions_table += f"<tr><td>{i + 1}</td><td>{label}</td><td>{score:.2f}</td></tr>"
        predictions_table += "</table>"
        st.markdown(predictions_table, unsafe_allow_html=True)

    except Exception as e:
        st.write("Error during classification:", str(e))



##
