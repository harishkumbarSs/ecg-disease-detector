# final_app.py

import streamlit as st
import numpy as np
from Ecg import ECG

# 1) Page config + hide menu/footer
st.set_page_config(
    page_title="ECG Disease Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 2) Sidebar
st.sidebar.title("ECG Disease Detector")
st.sidebar.write("Built with ❤️ and Streamlit")
uploaded = st.sidebar.file_uploader("Upload ECG Image", type=["png","jpg","jpeg"])
st.sidebar.markdown("---")
run_button = st.sidebar.button("Run Analysis")

# 3) Main
if uploaded and run_button:
    ecg = ECG()
    with st.spinner("Processing..."):
        # Load & preprocess
        img = ecg.getImage(uploaded)
        gray = ecg.GrayImage(img)
        leads = ecg.DividingLeads(img)
        ecg.PreprocessingLeads(leads)
        ecg.SignalExtraction_Scaling(leads)
        df_signal = ecg.CombineConvert1Dsignal()
        df_reduced = ecg.DimensionalReduciton(df_signal)
        prediction = ecg.ModelLoad_predict(df_reduced)

    # 4) Tabs for each step
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Upload & Preview",
        "2️⃣ Preprocessing",
        "3️⃣ Feature Extraction",
        "4️⃣ Prediction"
    ])

    with tab1:
        st.header("1. Upload & Preview")
        col1, col2 = st.columns(2)
        col1.subheader("Original ECG")
        col1.image(img, use_container_width=True)
        col2.subheader("Grayscale ECG")
        col2.image(gray, use_container_width=True)

    with tab2:
        st.header("2. Lead Segmentation & Preprocessing")
        st.subheader("Divided Leads (1–12)")
        st.image("Leads_1-12_figure.png", use_container_width=True)
        st.subheader("Long Lead (13)")
        st.image("Long_Lead_13_figure.png", use_container_width=True)
        st.markdown("----")
        st.subheader("Preprocessed Leads")
        st.image("Preprocessed_Leads_1-12_figure.png", use_container_width=True)
        st.image("Preprocessed_Leads_13_figure.png", use_container_width=True)

    with tab3:
        st.header("3. Signal Extraction & 1D Conversion")
        st.subheader("Extracted Signals")
        st.image("Contour_Leads_1-12_figure.png", use_container_width=True)
        st.markdown("----")
        st.subheader("1D Signal DataFrame")
        st.dataframe(df_signal, use_container_width=True)

        st.markdown("----")
        st.subheader("Dimensional Reduction (PCA Output)")
        st.dataframe(df_reduced, use_container_width=True)

    with tab4:
        st.header("4. Prediction")
        st.metric(label="Detected Condition", value=prediction)

else:
    st.markdown(
        "<div style='text-align:center; padding-top:50px;'>"
        "<h2>Welcome!</h2>"
        "<p>Please upload an ECG image<br>and click ‘Run Analysis’</p>"
        "</div>",
        unsafe_allow_html=True
    )
