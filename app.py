import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import zipfile
import os

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Title and description
st.title("AI Medical Report Analyzer Assistant")
st.write("""
Welcome to the AI Medical Report Analyzer Assistant! This tool helps healthcare professionals and patients
analyze medical reports by extracting key information and providing summaries.
""")

# Load dataset
@st.cache_data
def load_dataset():
    with zipfile.ZipFile('healthcare_dataset.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    df = pd.read_csv('healthcare_dataset.csv')
    return df

df = load_dataset()

# Sidebar options
st.sidebar.header("Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Full Report Summary", "Key Information Extraction", "Symptom Analysis"]
)

# Main content
st.header("Medical Report Analysis")

# Text input for medical report
medical_report = st.text_area("Enter Medical Report Text:", height=300)

if st.button("Analyze Report"):
    if medical_report:
        # Basic text processing
        words = word_tokenize(medical_report)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Generate basic summary (first 200 characters)
        summary = medical_report[:200] + "..." if len(medical_report) > 200 else medical_report
        
        # Display results based on analysis type
        if analysis_type == "Full Report Summary":
            st.subheader("Summary of Medical Report")
            st.write(summary)
        
        elif analysis_type == "Key Information Extraction":
            st.subheader("Extracted Medical Information")
            st.write("Key words found in the report:")
            for word in filtered_words[:10]:  # Show top 10 words
                st.write(f"- {word}")
        
        elif analysis_type == "Symptom Analysis":
            st.subheader("Identified Symptoms")
            symptoms = [word for word in filtered_words if word.lower() in ['pain', 'fever', 'cough', 'headache', 'nausea']]
            st.write("Symptoms found in the report:")
            for symptom in symptoms:
                st.write(f"- {symptom}")
    
    else:
        st.warning("Please enter a medical report to analyze")

# Display sample data
st.header("Sample Medical Data")
st.dataframe(df.head())
