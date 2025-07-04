import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

# =============== CONFIG & STYLING ==================
st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")

st.markdown("""
    <style>
    html, body, .main {
        background: linear-gradient(
            to bottom,
            #030509,
            #080c18,
            #0d1326,
            #121a35,
            #182243,
            #1d2952,
            #223060,
            #27376f,
            #27376f
        ) !important;
        background-attachment: fixed;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            to bottom,
            #f9fafd,
            #eaeff8,
            #cdd8ee,
            #92abdb,
            #839fd6,
            #7494d1,
            #6688cc,
            #577dc7,
            #4872c2
        ) !important;
    }

    .main * {
        color: white !important;
    }

    .stButton>button {
        background-color: #405dbc;
        color: white;
        border-radius: 8px;
    }

    div[data-baseweb="select"] > div,
    .stNumberInput input,
    input[type="text"],
    input[type="number"],
    textarea {
        background-color: white !important;
        border-radius: 6px !important;
        color: black !important;
    }

    div[data-baseweb="select"] * {
        color: black !important;
    }

    label {
        color: white !important;
    }

    .sidebar-bottom {
        margin-top: auto;
        padding-top: 20px;
        border-top: 1px solid #ccc;
        text-align: center;
    }

    .about-company {
        font-size: 20px;
        color: #ccc;
        padding: 10px;
        text-align: center;
    }

    .sidebar-bottom a {
        margin: 0 10px;
    }
            
    /* Force sidebar text to black */
    section[data-testid="stSidebar"] * {
        color: black !important;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        color: black !important;
    }
            

    section[data-testid="stSidebar"] .stRadio {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* Center-align the labels inside each radio option */
    section[data-testid="stSidebar"] .stRadio label {
        text-align: center;
        width: 100%;
    }
            
    </style>
""", unsafe_allow_html=True)

# =============== LOAD & CLEAN DATA ==================
@st.cache_data
def load_data():
    df = pd.read_csv(r"insurance_fraud_synthetic.csv")

    drop_cols = [col for col in df.columns if col.lower() in ['policy_number', 'incident_id', 'customer_id', 'claim_id']]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    df = df.dropna()
    df['fraud_reported'] = df['fraud_reported'].apply(lambda x: 1 if str(x).strip().upper() in ['Y', 'YES', '1'] else 0)

    nunique = df.nunique()
    df = df.drop(columns=nunique[nunique == 1].index)

    return df

df = load_data()

# =============== ENCODING & MODEL TRAINING ===============
@st.cache_data
def prepare_model(df):
    label_encoders = {}
    df_encoded = df.copy()

    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    target = 'fraud_reported'
    X = df_encoded.drop(target, axis=1)
    y = df_encoded[target]

    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return model, label_encoders, X, y, metrics, df_encoded

model, label_encoders, X, y, metrics, df_encoded = prepare_model(df)

# =============== SIDEBAR ==================
with st.sidebar:
    st.image("https://booleandata.com/wp-content/uploads/2022/09/Boolean-logo_Boolean-logo-USA-1-980x316.png", use_column_width=True)
    section = st.radio("", ["EDA", "Visualization", "ML Prediction"])
    st.markdown("---")
    st.markdown("""
        <div class='about-company'>
        <h5>🚀 About Us</h5>
        We are a data-driven company revolutionizing the insurance industry through predictive analytics. Our models help detect fraudulent claims with high accuracy and transparency.
        </div>
        <div class="sidebar-bottom">
          <a href="https://booleandata.ai/" target="_blank">🌐</a>
          <a href="https://www.facebook.com/Booleandata" target="_blank"><img src="https://cdn-icons-png.flaticon.com/24/1384/1384005.png" width="24"></a>
          <a href="https://www.youtube.com/channel/UCd4PC27NqQL5v9-1jvwKE2w" target="_blank"><img src="https://cdn-icons-png.flaticon.com/24/1384/1384060.png" width="24"></a>
          <a href="https://www.linkedin.com/company/boolean-data-systems" target="_blank"><img src="https://cdn-icons-png.flaticon.com/24/145/145807.png" width="24"></a>
        </div>
    """, unsafe_allow_html=True)

# =============== EDA ==================
if section == "EDA":
    st.title("📋 Exploratory Data Analysis")
    st.dataframe(df.head())
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    st.subheader("Fraud Class Distribution")

    with st.container():
        fig, ax = plt.subplots()

        # Set teal-green color palette: [Non-Fraud, Fraud]
        teal_green_palette = ['#66CDAA', '#008080']

        sns.countplot(data=df, x="fraud_reported", palette=teal_green_palette, ax=ax)

        # Optional styling
        ax.set_title("Fraud vs Non-Fraud Count", fontsize=14, color='black')
        ax.set_xlabel("Fraud Reported", color='black')
        ax.set_ylabel("Count", color='black')
        ax.tick_params(colors='black')

    fig.patch.set_facecolor('#f0f0f0')
    st.markdown("<div overflow:hidden; background-color:#f0f0f0; padding:10px;'>", unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("📈 Model Performance (Balanced Data)")
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    st.metric("Precision", f"{metrics['precision']:.2%}")
    st.metric("Recall", f"{metrics['recall']:.2%}")
    st.metric("F1 Score", f"{metrics['f1']:.2%}")

# =============== VISUALIZATION ==================
elif section == "Visualization":
    st.title("📊 Data Visualizations")

    st.subheader("Fraud vs Non-Fraud Distribution (Pie Chart)")
    with st.container():
        fig2, ax2 = plt.subplots(figsize=(8, 2))

        fraud_counts = df['fraud_reported'].value_counts()
        labels = ['Non-Fraud', 'Fraud']
        sizes = [fraud_counts[0], fraud_counts[1]]
        colors = ['#66CDAA', '#008080']  # Non-Fraud: light teal, Fraud: teal

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, textprops={'color': "black"})
        ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle

        fig.patch.set_facecolor('#f0f0f0')
        st.markdown("<div overflow:hidden; background-color:#f0f0f0; padding:10px;'>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
    import matplotlib.colors as mcolors

    st.subheader("Correlation Heatmap")
    with st.container():
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Create a custom teal-white colormap
        teal_white = mcolors.LinearSegmentedColormap.from_list("teal_white", ["#008080", "#ffffff"])

        sns.heatmap(df_encoded.corr(), annot=True, cmap=teal_white, fmt=".2f", ax=ax2)

        fig2.patch.set_facecolor('#f0f0f0')
        st.markdown("<div overflow:hidden; background-color:#f0f0f0; padding:10px;'>", unsafe_allow_html=True)
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Feature Importances (Line Chart)")
    with st.container():
        fig2, ax2 = plt.subplots(figsize=(8, 2))
        # Get top 10 features
        top_features = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(top_features.index, top_features.values, marker='o', linestyle='-', color='#008080', linewidth=2, markersize=8)

        # Style the chart
        ax.set_title("Top 10 Important Features", fontsize=14, color='black')
        ax.set_ylabel("Importance", fontsize=12, color='black')
        ax.set_xticklabels(top_features.index, rotation=45, ha='right', color='black')
        ax.tick_params(axis='y', colors='black')
        ax.grid(True, linestyle='--', alpha=0.4)

        fig.patch.set_facecolor('#f0f0f0')
        st.markdown("<div overflow:hidden; background-color:#f0f0f0; padding:10px;'>", unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
# =============== PREDICTION ==================
elif section == "ML Prediction":
    st.title("🔍 Insurance Fraud Detection")
    st.markdown("Fill the form to predict claim status:")

    user_input = {}
    use_test_case = st.checkbox("💡 Use Suspicious Example")

    with st.form("prediction_form"):
        for col in X.columns:
            if col in label_encoders:
                options = label_encoders[col].classes_
                default_val = options[0]
                if use_test_case:
                    fraud_case = {
                        "incident_type": "Collision",
                        "collision_type": "Rear Collision",
                        "incident_severity": "Major Damage",
                        "authorities_contacted": "None",
                        "insured_education_level": "High School",
                        "insured_occupation": "laborer",
                        "insured_relationship": "own-child",
                        "insured_sex": "MALE",
                        "auto_make": "Dodge",
                        "police_report_available": "NO"
                    }
                    default_val = fraud_case.get(col, options[0])

                user_input[col] = st.selectbox(f"{col}", options, index=options.tolist().index(default_val) if default_val in options else 0)

            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                default_val = mean_val
                if use_test_case:
                    fraud_nums = {
                        "incident_hour_of_the_day": 3,
                        "number_of_vehicles_involved": 3,
                        "witnesses": 0,
                        "total_claim_amount": 45000,
                        "injury_claim": 17000,
                        "property_claim": 10000,
                        "vehicle_claim": 8000,
                        "bodily_injuries": 2
                    }
                    default_val = fraud_nums.get(col, mean_val)

                user_input[col] = st.number_input(f"{col}", min_val, max_val, default_val)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_df = pd.DataFrame([user_input])
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.markdown(f"""
                <div style="background-color:#FFCDD2; padding: 20px; border-radius: 10px; border: 2px solid red;">
                    <h3 style="color:red;">⚠️ FRAUDULENT CLAIM DETECTED</h3>
                    <p style="font-size:18px;">🔍 Fraud Probability: <strong>{probability:.2%}</strong></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#C8E6C9; padding: 20px; border-radius: 10px; border: 2px solid green;">
                    <h3 style="color:green;">✅ CLAIM IS NON-FRAUDULENT</h3>
                    <p style="font-size:18px;">🔍 Confidence: <strong>{(1 - probability):.2%}</strong></p>
                </div>
            """, unsafe_allow_html=True)
