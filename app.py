import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the dataset (ensure the CSV is in the same directory)
data = pd.read_csv('Filtered_Neuro_Disorders_Data.csv')

# Encode the 'Disease' labels
label_encoder = LabelEncoder()
data['disease_encoded'] = label_encoder.fit_transform(data['Disease'])
num_classes = len(label_encoder.classes_)

# Symptoms dataset with detailed and complex symptoms for 4 diseases
symptoms_data = {
    "Cerebral Palsy": [
        "My child has difficulty walking, their legs seem stiff, and they fall frequently.",
        "There is involuntary jerking in the arms, and they cannot hold objects for long periods.",
        "Trouble controlling body movements, poor posture, and muscles feel tight all the time.",
        "They have delays in motor skills like crawling, sitting, or rolling over."
    ],
    "Duchenne Muscular Dystrophy": [
        "I feel extreme muscle weakness in my legs, climbing stairs is very difficult.",
        "There’s noticeable calf enlargement and frequent fatigue even after mild activities.",
        "I struggle with standing up from a seated position and keeping balance while walking.",
        "Progressive weakness over months in the arms and chest muscles, making it hard to lift objects."
    ],
    "Giant Axonal Neuropathy": [
        "My child has uncoordinated walking and frequent falls. Their hair looks abnormally curly.",
        "Muscle stiffness, weakness in hands, and difficulty swallowing or speaking clearly.",
        "They experience tremors, loss of sensation in fingers and toes, and extreme fatigue.",
        "Rapid deterioration in motor skills and inability to grip objects tightly or control hand movements."
    ],
    "Hydranencephaly": [
        "My baby has a large head circumference and difficulty moving their limbs voluntarily.",
        "They show minimal response to visual stimuli, and their cry sounds weak or unusual.",
        "The child has seizures, poor feeding ability, and developmental delays like not sitting or rolling over.",
        "There seems to be excessive fluid in the head, and the baby seems unaware of their surroundings."
    ]
}

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define function to preprocess input text (tokenize and pad)
def preprocess_input(text, tokenizer, max_length=128):
    encoded = tokenizer.encode_plus(
        text, 
        add_special_tokens=True, 
        max_length=max_length, 
        padding='max_length', 
        truncation=True, 
        return_tensors='np'
    )
    return encoded['input_ids'], encoded['attention_mask']

# Load the trained model (assuming you've trained the model or loaded the weights)
# Initialize the BERT-based model
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

# BERT model as the base
bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]

# Add a Dense layer on top of the BERT output
dense_layer = tf.keras.layers.Dense(128, activation='relu')(bert_output)

# Output layer for classification (number of classes = number of unique diseases)
output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dense_layer)

# Compile the model
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Here, load your pre-trained model weights (after training)
# model.load_weights('path_to_your_model_weights.h5')

# Function to predict disease based on symptoms
def predict_disease_from_symptoms(symptoms):
    combined_text = " ".join(symptoms)  # Combine symptoms into a single text
    
    input_ids, attention_mask = preprocess_input(combined_text, tokenizer)
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    # Predict disease
    prediction = model.predict(inputs)
    predicted_class = np.argmax(prediction, axis=1)
    confidence_level = np.max(prediction, axis=1)
    predicted_disease = label_encoder.inverse_transform(predicted_class)
    
    result = {
        "Predicted Disease": predicted_disease[0],
        "Confidence Level": confidence_level[0] * 100  # Convert to percentage
    }
    
    return result

# Streamlit UI
st.set_page_config(page_title="Neuro Disease Prediction", page_icon="🧠", layout="centered")

st.markdown(
    """
    # Genetic Disorder Prediction System 🧠
    Predict potential genetic disorder based on symptoms you provide.
    This application uses a **SNGP-BERT-based model** to analyze text input and make predictions.
    """, 
    unsafe_allow_html=True
)

# Input for user symptoms
st.write("### Enter Symptoms")
st.write("Please describe your symptoms below. You can list multiple symptoms, separated by commas.")

user_symptoms = st.text_area("Symptoms", placeholder="e.g. I feel weak, I can't walk straight...", height=150)

# Add a Submit button
if st.button("Submit"):
    if user_symptoms.strip():
        with st.spinner("Predicting disease..."):
            symptoms_list = [symptom.strip() for symptom in user_symptoms.split(",")]

            prediction = predict_disease_from_symptoms(symptoms_list)

            st.markdown(
                f"""
                ## 🏷️ **Prediction Result**:
                - **Disease**: {prediction['Predicted Disease']}
                - **Confidence Level**: {prediction['Confidence Level']:.2f}%
                """, 
                unsafe_allow_html=True
            )

            
    else:
        st.error("Please enter some symptoms before submitting!")

st.markdown(
    """
    ---
    **Developed by Nidhi Shakhapur (21BCB0194)**  
    Powered by TensorFlow and Hugging Face Transformers.  
    """, 
    unsafe_allow_html=True
)
