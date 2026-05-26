import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

# ------------------------------------------------
# Page Config
# ------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------------------------
# Custom CSS
# ------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #F8F9F9;
}

.title {
    text-align:center;
    font-size:40px;
    color:#C0392B;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:gray;
    font-size:18px;
}

.card {
    background-color:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
}

.prediction-good {
    background:#D5F5E3;
    padding:20px;
    border-radius:10px;
    color:#145A32;
    font-size:22px;
    text-align:center;
    font-weight:bold;
}

.prediction-bad {
    background:#FADBD8;
    padding:20px;
    border-radius:10px;
    color:#922B21;
    font-size:22px;
    text-align:center;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load Model
# ------------------------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Dataset
data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# ------------------------------------------------
# Title
# ------------------------------------------------
st.markdown(
    "<div class='title'>🩺 Breast Cancer Detection using SVM</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning powered tumor classification dashboard</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966487.png",
    width=120
)

st.sidebar.header("Tumor Measurements")

mean_radius = st.sidebar.slider(
    "Mean Radius", 5.0,30.0,14.0
)

mean_texture = st.sidebar.slider(
    "Mean Texture", 5.0,40.0,20.0
)

mean_perimeter = st.sidebar.slider(
    "Mean Perimeter", 40.0,200.0,90.0
)

mean_area = st.sidebar.slider(
    "Mean Area", 100.0,2500.0,600.0
)

mean_smoothness = st.sidebar.slider(
    "Mean Smoothness", 0.05,0.20,0.10
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Benign → Non-cancerous

    Malignant → Cancerous
    """
)

# ------------------------------------------------
# Metrics
# ------------------------------------------------
m1,m2,m3 = st.columns(3)

m1.metric(
    "Dataset Samples",
    len(df)
)

m2.metric(
    "Features",
    30
)

m3.metric(
    "Model Accuracy",
    "98%"
)

st.markdown("---")

# ------------------------------------------------
# Prediction Layout
# ------------------------------------------------
col1,col2 = st.columns([1,1])

with col1:

    st.subheader("Input Summary")

    input_df = pd.DataFrame({
        'Feature':[
            'Radius',
            'Texture',
            'Perimeter',
            'Area',
            'Smoothness'
        ],
        'Value':[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness
        ]
    })

    st.dataframe(
        input_df,
        use_container_width=True
    )

with col2:

    st.subheader("Prediction")

    if st.button("Predict Tumor Type"):

        full_input = np.zeros((1,30))

        full_input[0][0]=mean_radius
        full_input[0][1]=mean_texture
        full_input[0][2]=mean_perimeter
        full_input[0][3]=mean_area
        full_input[0][4]=mean_smoothness

        scaled = scaler.transform(
            full_input
        )

        prediction = model.predict(
            scaled
        )

        prob = model.predict_proba(
            scaled
        )

        malignant_prob = prob[0][0]*100

        st.progress(
            int(malignant_prob)
        )

        if prediction[0]==0:

            st.markdown(
                f"""
                <div class='prediction-bad'>
                ⚠️ Malignant Tumor
                <br>
                Risk: {malignant_prob:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class='prediction-good'>
                ✅ Benign Tumor
                <br>
                Confidence: {100-malignant_prob:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # Probability chart
        fig,ax = plt.subplots()

        ax.bar(
            ["Malignant","Benign"],
            prob[0]
        )

        ax.set_ylabel(
            "Probability"
        )

        ax.set_title(
            "Prediction Probability"
        )

        st.pyplot(fig)

# ------------------------------------------------
# Dashboard
# ------------------------------------------------
st.markdown("---")
st.header("Dataset Dashboard")

g1,g2 = st.columns(2)

# Pie chart
with g1:

    st.subheader("Tumor Distribution")

    fig1,ax1 = plt.subplots()

    df_target = pd.Series(
        data.target
    ).value_counts()

    ax1.pie(
        df_target,
        labels=[
            "Malignant",
            "Benign"
        ],
        autopct='%1.1f%%'
    )

    st.pyplot(fig1)

# Heatmap
with g2:

    st.subheader("Correlation Heatmap")

    fig2,ax2 = plt.subplots(
        figsize=(7,5)
    )

    sns.heatmap(
        df.corr(),
        cmap='coolwarm',
        ax=ax2
    )

    st.pyplot(fig2)

# Scatter plot
st.subheader("Radius vs Texture")

fig3,ax3 = plt.subplots()

sns.scatterplot(
    x=df['mean radius'],
    y=df['mean texture'],
    hue=data.target,
    ax=ax3
)

st.pyplot(fig3)

st.markdown("---")

st.caption(
    "Built with Streamlit • SVM • Scikit-Learn"
)