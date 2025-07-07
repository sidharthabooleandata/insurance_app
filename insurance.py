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
        background-color: #EBEEF8 !important;
        background: #EBEEF8 !important;
        color: black;
    }

    /* Sidebar Gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            to bottom,
            #f9fafd,
            #e6eaf0,
            #c5cbd6,
            #9fa6b3,
            #7b8190,
            #5c6374,
            #3f4556,
            #232739,
            #161925,
            #0c0e14,
            #030509
        ) !important;
    }

    /* Main Content Headers */
    .main h1, .main h2, .main h3, .main p, .main label {
        color: #1c2953;
    }

    /* Buttons */
    .stButton>button {
        background-color: #405dbc;
        color: white;
        border-radius: 8px;
    }

    /* Inputs */
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
        color: black !important;
    }

    /* Sidebar Bottom Section */
    .sidebar-bottom {
        margin-top: auto;
        padding-top: 20px;
        border-top: 1px solid #ccc;
        text-align: center;
    }

    /* ✅ About Us Text - White */
    .about-company {
        font-size: 15px;
        color: #dce2f3 !important;
        padding: 10px;
        text-align: center;
    }

    /* ✅ Social Media Icons Highlighted */
    .sidebar-bottom a img {
        filter: drop-shadow(0 0 2px white) brightness(1.2);
        transition: transform 0.2s ease;
        margin: 0 8px;
    }
    .sidebar-bottom a img:hover {
        transform: scale(1.1);
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
    section[data-testid="stSidebar"] .stRadio label {
        text-align: center;
        width: 100%;
    }

    /* Prediction Results */
    .fraud-text {
        color: darkred !important;
        font-size: 18px;
        font-weight: bold;
    }
    .nonfraud-text {
        color: darkgreen !important;
        font-size: 18px;
        font-weight: bold;
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
    section = st.radio("", [ "Visualization", "ML Prediction"])
    st.markdown("---")
    st.markdown("""
        <div class='about-company'>
        <h1>🚀 About Us</h1>
        We are a data-driven company revolutionizing the insurance industry through predictive analytics. Our models help detect fraudulent claims with high accuracy and transparency.These solutions lower costs and enhance output, designed to transform smoothly as your enterprise.
        
        </div>
        <div class="sidebar-bottom">
          <a href="https://booleandata.ai/" target="_blank">🌐</a>
          <a href="https://www.facebook.com/Booleandata" target="_blank"><img src="https://cdn-icons-png.flaticon.com/24/1384/1384005.png" width="24"></a>
          <a href="https://www.youtube.com/channel/UCd4PC27NqQL5v9-1jvwKE2w" target="_blank"><img src="https://cdn-icons-png.flaticon.com/24/1384/1384060.png" width="24"></a>
          <a href="https://www.linkedin.com/company/boolean-data-systems" target="_blank"><img src="https://cdn-icons-png.flaticon.com/24/145/145807.png" width="24"></a>
        </div>
    """, unsafe_allow_html=True)


# =============== VISUALIZATION ==================
if section == "Visualization":
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load and preprocess data
    @st.cache_data
    def load_data():
        df = pd.read_csv("insurance_fraud_synthetic.csv")
        df['fraud_reported'] = df['fraud_reported'].apply(lambda x: 1 if str(x).strip().upper() in ['Y', 'YES', '1'] else 0)
        df['property_damage'] = df['property_damage'].fillna("Unknown")
        df['collision_type'] = df['collision_type'].fillna("Unknown")
        df['police_report_available'] = df['police_report_available'].fillna("Unknown")
        return df

    df = load_data()
    st.title("📊 Insurance Fraud Analysis Dashboard")

    # ========== BAR CHART ==========
    st.markdown("### 🔷 Filter for Bar Chart")
    bar_col1, bar_col2 = st.columns(2)
    with bar_col1:
        x_bar = st.selectbox("Bar Chart X-axis (Categorical)", df.select_dtypes(include='object').columns, key="bar_x")
    with bar_col2:
        y_bar = st.selectbox("Bar Chart Y-axis (Numerical)", df.select_dtypes(include='number').columns, key="bar_y")

    bar_data = df.groupby(x_bar)[y_bar].mean().reset_index()
    bar_colors = px.colors.sequential.Blues + px.colors.sequential.PuBuGn[::-1]  # Mixed blue palette

    fig_bar = px.bar(bar_data, x=x_bar, y=y_bar, text_auto='.2s', title=f"Avg {y_bar} by {x_bar}")
    fig_bar.update_traces(marker=dict(line=dict(color='black', width=1), color=bar_data[y_bar], colorscale='Blues'))
    fig_bar.update_layout(template="plotly_white", plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)

    # ========== LINE CHART ==========
    st.markdown("### 🔷 Filter for Line Chart")
    line_col1, line_col2 = st.columns(2)
    with line_col1:
        x_line = st.selectbox("Line Chart X-axis (Categorical)", df.select_dtypes(include='object').columns, key="line_x")
    with line_col2:
        y_line = st.selectbox("Line Chart Y-axis (Numerical)", df.select_dtypes(include='number').columns, key="line_y")

    line_data = df.groupby(x_line)[y_line].mean().reset_index()
    fig_line = px.line(line_data, x=x_line, y=y_line, title=f"Trend of {y_line} over {x_line}",
                    markers=True, line_shape="spline", color_discrete_sequence=["#002060"])
    fig_line.update_traces(line=dict(width=3))
    fig_line.update_layout(template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True)

    # ========== PIE & DONUT ==========
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🍩 Donut Chart - Fraud Reported")
        fig_donut = px.pie(df, names='fraud_reported', hole=0.5, title="Fraud vs Non-Fraud",
                        color_discrete_sequence=px.colors.sequential.PuBu[::-1])
        fig_donut.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("#### 🥧 Pie Chart - Incident Type")
        fig_pie = px.pie(df, names='incident_type', title="Incident Type Distribution",
                        color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

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
                    <h3 class="fraud-text">⚠️ FRAUDULENT CLAIM DETECTED</h3>
                    <p style="all: unset; color: black; font-size: 18px; font-weight: bold;">
                        🔍 Fraud Probability: <strong style="all: unset; color: black;">{probability:.2%}</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#C8E6C9; padding: 20px; border-radius: 10px; border: 2px solid green;">
                    <h3 class="nonfraud-text">✅ CLAIM IS NON-FRAUDULENT</h3>
                    <p style="all: unset; color: black; font-size: 18px; font-weight: bold;">
                        🔍 Confidence: <strong style="all: unset; color: black;">{(1 - probability):.2%}</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)
