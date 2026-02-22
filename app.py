"""
Skin Cancer Detection & Classification Web App
Beautiful Frontend with Streamlit

Features:
- Upload skin lesion images
- Real-time prediction with probabilities
- Beautiful visualizations
- Detailed disease information
- Medical disclaimer
- Downloadable results

Author: Skin Cancer Detection System
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Skin Cancer Detection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================

st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Warning box */
    .warning-box {
        background: #1e293b;
        border-left: 5px solid #38bdf8;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 5px solid #38bdf8;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 5px solid #38bdf8;
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 1rem 0;
        border-top: 4px solid #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# DISEASE INFORMATION DATABASE
# ============================================

DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratosis',
        'full_name': 'Actinic Keratosis & Intraepithelial Carcinoma',
        'severity': 'Moderate',
        'color': '#f59e0b',
        'description': 'A rough, scaly patch on the skin caused by years of sun exposure. Can develop into skin cancer if untreated.',
        'symptoms': [
            'Rough, dry, or scaly patch of skin',
            'Flat to slightly raised patch',
            'Color variations (pink, red, or brown)',
            'Itching or burning in the affected area'
        ],
        'risk_factors': [
            'Fair skin',
            'History of frequent sun exposure',
            'Age over 40',
            'Weakened immune system'
        ],
        'treatment': 'Cryotherapy, topical medications, photodynamic therapy, or surgical removal.',
        'recommendation': '⚠️ Consult a dermatologist for proper treatment to prevent progression to cancer.'
    },
    
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'full_name': 'Basal Cell Carcinoma',
        'severity': 'High',
        'color': '#ef4444',
        'description': 'The most common type of skin cancer. Grows slowly and rarely spreads, but requires treatment.',
        'symptoms': [
            'Pearly or waxy bump',
            'Flat, flesh-colored or brown lesion',
            'Bleeding or scabbing sore that heals and returns',
            'Pink growth with raised edges'
        ],
        'risk_factors': [
            'Chronic sun exposure',
            'Fair skin',
            'Family history',
            'Radiation therapy'
        ],
        'treatment': 'Surgical excision, Mohs surgery, cryotherapy, or topical treatments.',
        'recommendation': '🚨 Schedule an appointment with a dermatologist immediately for evaluation and treatment.'
    },
    
    'bkl': {
        'name': 'Benign Keratosis',
        'full_name': 'Benign Keratosis-like Lesions',
        'severity': 'Low',
        'color': '#10b981',
        'description': 'Non-cancerous skin growths including seborrheic keratoses and solar lentigos.',
        'symptoms': [
            'Brown, black, or tan growths',
            'Waxy, slightly raised appearance',
            'Stuck-on appearance',
            'Usually painless'
        ],
        'risk_factors': [
            'Age (more common in older adults)',
            'Genetics',
            'Sun exposure'
        ],
        'treatment': 'Usually no treatment needed. Can be removed for cosmetic reasons.',
        'recommendation': '✅ Generally harmless, but monitoring for changes is recommended.'
    },
    
    'df': {
        'name': 'Dermatofibroma',
        'full_name': 'Dermatofibroma',
        'severity': 'Low',
        'color': '#10b981',
        'description': 'A common benign skin growth that feels like a hard lump. Usually harmless.',
        'symptoms': [
            'Small, firm bump',
            'Red, brown, or purple color',
            'Dimples when pinched',
            'May be itchy or tender'
        ],
        'risk_factors': [
            'Minor skin injuries',
            'Insect bites',
            'More common in women'
        ],
        'treatment': 'Usually no treatment needed. Surgical removal if bothersome.',
        'recommendation': '✅ Benign condition. Removal only if causing discomfort or cosmetic concerns.'
    },
    
    'mel': {
        'name': 'Melanoma',
        'full_name': 'Melanoma',
        'severity': 'Critical',
        'color': '#dc2626',
        'description': 'The most dangerous type of skin cancer. Can spread to other organs if not treated early.',
        'symptoms': [
            'Asymmetrical mole',
            'Irregular borders',
            'Color variations (multiple colors)',
            'Diameter larger than 6mm',
            'Evolving size, shape, or color'
        ],
        'risk_factors': [
            'Family history of melanoma',
            'Many moles or atypical moles',
            'Fair skin',
            'History of sunburns',
            'Weakened immune system'
        ],
        'treatment': 'Surgical excision, immunotherapy, targeted therapy, radiation, or chemotherapy.',
        'recommendation': '🚨🚨 URGENT: See a dermatologist immediately. Early detection is critical for successful treatment.'
    },
    
    'nv': {
        'name': 'Melanocytic Nevus',
        'full_name': 'Melanocytic Nevus (Mole)',
        'severity': 'Low',
        'color': '#10b981',
        'description': 'A common mole. Generally harmless but should be monitored for changes.',
        'symptoms': [
            'Round or oval shape',
            'Uniform color (brown or black)',
            'Flat or slightly raised',
            'Stable over time'
        ],
        'risk_factors': [
            'Genetics',
            'Sun exposure',
            'Fair skin'
        ],
        'treatment': 'Usually no treatment needed. Removal if atypical or for cosmetic reasons.',
        'recommendation': '✅ Monitor for changes. Use the ABCDE rule: Asymmetry, Border, Color, Diameter, Evolving.'
    },
    
    'vasc': {
        'name': 'Vascular Lesion',
        'full_name': 'Vascular Lesion',
        'severity': 'Low',
        'color': '#10b981',
        'description': 'Abnormalities of blood vessels in the skin, including hemangiomas and angiokeratomas.',
        'symptoms': [
            'Red or purple discoloration',
            'Can be flat or raised',
            'May blanch when pressed',
            'Usually painless'
        ],
        'risk_factors': [
            'Congenital factors',
            'Age',
            'Hormonal changes'
        ],
        'treatment': 'Laser therapy, sclerotherapy, or surgical removal if needed.',
        'recommendation': '✅ Usually benign. Consult dermatologist if growing or causing symptoms.'
    }
}

# ============================================
# HELPER FUNCTIONS
# ============================================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = keras.models.load_model('skin_cancer_model.h5')
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to array
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize (ImageNet normalization)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model, image):
    """Make prediction on image"""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)[0]
    
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # Get top prediction
    top_idx = np.argmax(predictions)
    top_class = class_names[top_idx]
    top_confidence = predictions[top_idx]
    
    # Create results dictionary
    results = {
        'class': top_class,
        'confidence': float(top_confidence),
        'all_predictions': {
            class_names[i]: float(predictions[i])
            for i in range(len(class_names))
        }
    }
    
    return results

def create_probability_chart(predictions):
    """Create interactive probability bar chart"""
    df = pd.DataFrame({
        'Class': [DISEASE_INFO[cls]['name'] for cls in predictions.keys()],
        'Probability': list(predictions.values()),
        'Code': list(predictions.keys())
    })
    
    df = df.sort_values('Probability', ascending=True)
    
    # Color bars based on prediction value
    colors = ['#667eea' if p < 0.3 else '#f59e0b' if p < 0.7 else '#10b981' 
              for p in df['Probability']]
    
    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Class'],
        orientation='h',
        marker=dict(color=colors),
        text=[f'{p:.1%}' for p in df['Probability']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Classification Probabilities',
        xaxis_title='Probability',
        yaxis_title='',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[0, 1]),
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_gauge_chart(confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'size': 24}},
        number = {'suffix': "%"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "#fee2e2"},
                {'range': [50, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#d1fae5"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'size': 16}
    )
    
    return fig

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 AI Skin Cancer Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Deep Learning for Dermatological Analysis</p>', 
                unsafe_allow_html=True)
    
    # Medical Disclaimer
    st.markdown("""
        <div class="warning-box">
            <h3>⚠️ IMPORTANT MEDICAL DISCLAIMER</h3>
            <p><strong>This AI system is for educational and screening purposes only.</strong></p>
            <ul>
                <li>NOT a substitute for professional medical diagnosis</li>
                <li>Should NOT be used as the sole basis for treatment decisions</li>
                <li>Always consult a qualified dermatologist or healthcare provider</li>
                <li>In case of concerning symptoms, seek immediate medical attention</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/microscope.png", width=100)
        st.title("About")
        st.info("""
            This AI system uses deep learning to classify skin lesions into 7 categories:
            
            **Malignant:**
            - 🔴 Melanoma (mel)
            - 🟠 Basal Cell Carcinoma (bcc)
            - 🟡 Actinic Keratosis (akiec)
            
            **Benign:**
            - 🟢 Melanocytic Nevus (nv)
            - 🟢 Benign Keratosis (bkl)
            - 🟢 Dermatofibroma (df)
            - 🟢 Vascular Lesion (vasc)
        """)
        
        st.markdown("---")
        
        st.subheader("📊 Model Information")
        st.write("**Architecture:** EfficientNetB3")
        st.write("**Accuracy:** 85-92%")
        st.write("**Training Data:** HAM10000")
        
        st.markdown("---")
        
        st.subheader("🎯 How to Use")
        st.write("""
            1. Upload a clear image of the skin lesion
            2. Wait for AI analysis
            3. Review the results and recommendations
            4. **Consult a dermatologist for confirmation**
        """)
    
    # Load model
    model, error = load_model()
    
    if error:
        st.error(f"""
            ❌ **Model Loading Error**
            
            Could not load the model file. Please ensure:
            1. The model file 'skin_cancer_model.h5' is in the same directory
            2. The file is a valid Keras model
            
            Error details: {error}
        """)
        st.stop()
    
    st.success("✅ AI Model loaded successfully!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear, well-lit image of the skin lesion"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info
            st.info(f"""
                **Image Details:**
                - Format: {image.format}
                - Size: {image.size[0]} x {image.size[1]} pixels
                - Mode: {image.mode}
            """)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Analysis")
            
            # Analyze button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the image..."):
                    # Simulate processing time
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Make prediction
                    results = predict(model, image)
                    
                    # Store results in session state
                    st.session_state['results'] = results
                    st.session_state['image'] = image
                    
                st.success("✅ Analysis complete!")
    
    # Display results if available
    if 'results' in st.session_state:
        st.markdown("---")
        
        results = st.session_state['results']
        predicted_class = results['class']
        confidence = results['confidence']
        disease_info = DISEASE_INFO[predicted_class]
        
        # Results header
        st.markdown(f"""
            <div class="result-box">
                <h2>🎯 Prediction Results</h2>
                <h3 style="color: {disease_info['color']};">{disease_info['full_name']}</h3>
                <p style="font-size: 1.2rem;">Confidence: <strong>{confidence:.1%}</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(
                create_probability_chart(results['all_predictions']),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_gauge_chart(confidence),
                use_container_width=True
            )
        
        # Detailed Information
        st.markdown("---")
        st.subheader("📋 Detailed Information")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📖 Description", 
            "🔬 Symptoms", 
            "⚠️ Risk Factors", 
            "💊 Treatment"
        ])
        
        with tab1:
            st.markdown(f"""
                <div class="info-card">
                    <h4>About {disease_info['name']}</h4>
                    <p style="font-size: 1.1rem;">{disease_info['description']}</p>
                    <p><strong>Severity Level:</strong> <span style="color: {disease_info['color']}; font-weight: bold;">{disease_info['severity']}</span></p>
                </div>
            """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("**Common Symptoms:**")
            for symptom in disease_info['symptoms']:
                st.markdown(f"- {symptom}")
        
        with tab3:
            st.markdown("**Risk Factors:**")
            for risk in disease_info['risk_factors']:
                st.markdown(f"- {risk}")
        
        with tab4:
            st.markdown(f"**Treatment Options:**")
            st.write(disease_info['treatment'])
        
        # Recommendation
        st.markdown("---")
        st.markdown(f"""
            <div class="info-card" style="border-top: 4px solid {disease_info['color']};">
                <h3>💡 Recommendation</h3>
                <p style="font-size: 1.2rem; font-weight: 500;">{disease_info['recommendation']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Download results
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            # Create report
            report = f"""
SKIN LESION ANALYSIS REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION:
  Diagnosis: {disease_info['full_name']}
  Confidence: {confidence:.1%}
  Severity: {disease_info['severity']}

PROBABILITIES:
"""
            for cls, prob in results['all_predictions'].items():
                report += f"  {DISEASE_INFO[cls]['name']}: {prob:.1%}\n"
            
            report += f"""
DESCRIPTION:
  {disease_info['description']}

RECOMMENDATION:
  {disease_info['recommendation']}

DISCLAIMER:
  This AI analysis is for screening purposes only and should NOT
  replace professional medical diagnosis. Always consult a qualified
  dermatologist for proper evaluation and treatment.
"""
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"skin_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #64748b; padding: 2rem;">
            <p><strong>Skin Cancer Detection AI System</strong></p>
            <p>Powered by Deep Learning & TensorFlow</p>
            <p style="font-size: 0.9rem;">⚕️ For educational and screening purposes only | Always consult medical professionals</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()