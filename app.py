import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# Configure page
st.set_page_config(
    page_title="DDoS Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Alan+Sans:wght@300..900&display=swap');
    
    * {
        font-family: "Alan Sans", sans-serif;
    }
    
    /* Hide ALL Streamlit branding and menu items */
    #MainMenu {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    
    /* Hide GitHub icon and deploy button */
    .stDeployButton {display: none !important;}
    button[kind="header"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    
    /* Hide the entire header bar */
    header[data-testid="stHeader"] {display: none !important;}
    
    /* Main background */
    .stApp {
        background-color: #e5e7eb;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #065f46 0%, #047857 100%);
        padding: 2.5rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
        line-height: 1.8;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    [data-testid="stSidebar"] h2 {
        font-size: 1.75rem;
        margin-bottom: 2rem;
    }
    
    [data-testid="stSidebar"] h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] li {
        font-size: 1.2rem;
        line-height: 2;
    }
    
    [data-testid="stSidebar"] strong {
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
        margin: 1.5rem 0;
    }
    
    /* Main content container */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 4rem;
        margin: 0 auto;
    }
    
    /* Welcome card */
    .welcome-card {
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2.5rem;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #065f46;
        margin-bottom: 1.5rem;
        letter-spacing: -0.03em;
    }
    
    .welcome-text {
        font-size: 1.5rem;
        color: black;
        line-height: 2;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Dashboard cards */
    .dashboard-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        height: 100%;
        margin-bottom:1rem;
    }
    
    .card-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        letter-spacing: -0.02em;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
        min-height: 180px;
        display: flex;
        align-items: stretch;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 1.5rem;
        opacity: 0.95;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(6, 95, 70, 0.4);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.5rem;
        width: 100%;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        font-size: 1.5rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.75rem 1rem;
    }
    
    /* Loader */
    .loader-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
    }
    
    .loader {
        border: 8px solid rgba(255,255,255,0.2);
        border-radius: 50%;
        border-top: 8px solid white;
        width: 80px;
        height: 80px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loader-text {
        color: white;
        font-size: 1.5rem;
        margin-top: 2rem;
        font-weight: 600;
    }
     /* Kill all toolbar/action buttons */
    [data-testid="stToolbar"],
    [data-testid="stActionButton"],
    button[kind="header"],
    .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        /* Extra hiding for mobile */
        #MainMenu, footer, header, 
        .stDeployButton, 
        button[kind="header"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        header[data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
        }
        
        .main .block-container {
            padding: 1rem;
        }
        
        .welcome-title {
            font-size: 2rem;
        }
        
        .welcome-text {
            font-size: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        [data-testid="stSidebar"] h2 {
            font-size: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        models = {
            'xgb_binary': joblib.load('xgb_binary_model.pkl'),
            'xgb_attack': joblib.load('xgb_attack_model.pkl'),
            'scaler': joblib.load('scaler.pkl'),
            'scaler_attack': joblib.load('scaler_attack.pkl'),
            'label_encoder': joblib.load('label_encoder.pkl'),
            'feature_names': joblib.load('feature_names.pkl')
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None

def show_loader():
    """Display professional rotating loader"""
    loader_html = """
    <div class="loader-container">
        <div class="loader"></div>
        <div class="loader-text">Loading DDoS Detection System...</div>
    </div>
    """
    loader_placeholder = st.empty()
    loader_placeholder.markdown(loader_html, unsafe_allow_html=True)
    time.sleep(3)
    loader_placeholder.empty()

def predict_traffic(data, models):
    """Predict traffic using XGBoost with two-stage classification"""
    try:
        expected_features = models['feature_names']
        X = data.drop(columns=["label", "activity"], errors='ignore')
        
        missing_features = set(expected_features) - set(X.columns)
        if missing_features:
            st.error(f"Missing required features: {list(missing_features)[:10]}...")
            return None,
        
        X = X[expected_features]
        X_scaled = models['scaler'].transform(X)
        
        binary_pred = models['xgb_binary'].predict(X_scaled)
        binary_proba = models['xgb_binary'].predict_proba(X_scaled)
        
        attack_indices = binary_pred == 1
        attack_predictions = []
        
        if attack_indices.any():
            X_attack_scaled = models['scaler_attack'].transform(X[attack_indices])
            attack_pred_encoded = models['xgb_attack'].predict(X_attack_scaled)
            attack_predictions = models['label_encoder'].inverse_transform(attack_pred_encoded)
        
        return binary_pred, binary_proba, attack_predictions, attack_indices
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None, None

def main():
    # Show loader on first visit
    if 'loaded' not in st.session_state:
        show_loader()
        st.session_state.loaded = True
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: Left; margin-bottom: 2rem;'>🛡️ DDoS Detector</h2>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Model Information")
        
        models = load_models()
        if models:
            st.markdown(f"""
            **Algorithm:** XGBoost  
            **Features:** {len(models['feature_names'])}  
            **Accuracy:** 99%  
            **Attack Types:** {len(models['label_encoder'].classes_)}
            """)
        
        st.markdown("---")
        st.markdown("### Navigation")
        st.markdown("📊 Dashboard")
        st.markdown("🔍 Analysis")
        st.markdown("📥 Results")
        
        st.markdown("---")
        
        with st.expander("Attack Types Detected"):
            if models:
                for attack_type in models['label_encoder'].classes_:
                    st.markdown(f"• {attack_type}")
        
        with st.expander("Feature Categories"):
            st.markdown("""
            • Port Information
            • Packet Statistics  
            • Payload Metrics
            • Timing Features
            • TCP Flags
            • Header Information
            """)
    
    if models is None:
        st.stop()
    
    # ROW 1: Welcome Section
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">DDoS Detection and Classification System</div>
            <div class="welcome-text">
                Protect your networks and websites from Distributed Denial of Service attacks 
                using advanced machine learning algorithms. Our system analyzes network traffic patterns 
                in real-time to detect and classify potential threats with 99% accuracy.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ROW 2: Upload and Info Section
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('''
            <div class="dashboard-card">
                <div class="card-title">📁 Upload Network Traffic</div>
            </div>
        ''', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose CSV file containing network traffic data",
            type="csv",
            help="File must contain 317 network traffic features"
        )
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {data.shape[0]} samples with {data.shape[1]} features")
            
            with st.expander("Preview Data"):
                st.dataframe(data.head(), use_container_width=True)
    
    with col2:
        st.markdown('''
            <div class="dashboard-card">
                <div class="card-title">ℹ️ How It Works</div>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("""
        **Step 1:** Upload your network traffic CSV file  
        **Step 2:** Click the Analyze button  
        **Step 3:** View detection results and attack classifications  
        **Step 4:** Download detailed analysis report
        
        ---
        
        **Supported Formats:**  
        • CSV files with 317 features  
        • Maximum file size: 200MB
        """)
    
    # ROW 3: Analysis and Results Section
    if uploaded_file:
        st.markdown('''
            <div class="result-card">
                <div class="card-title">🔍 Traffic Analysis</div>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Analyze Traffic Now"):
            with st.spinner("Analyzing network traffic patterns..."):
                binary_pred, binary_proba, attack_pred, attack_indices = predict_traffic(data, models)
                
                if binary_pred is not None:
                    benign_count = sum(binary_pred == 0)
                    attack_count = sum(binary_pred == 1)
                    
                    # Metrics
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    with col_m1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Total Samples</div>
                                <div class="metric-value">{len(data)}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m2:
                        st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%);">
                                <div class="metric-label">Benign Traffic</div>
                                  <div class="metric-value">{benign_count} <span style="font-size: 1.75rem; font-weight: 500;">({benign_count/len(data)*100:.1f}%)</span></div>

                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m3:
                        st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                                <div class="metric-label">Attack Traffic</div>
                                <div class="metric-value">{attack_count} <span style="font-size: 1.75rem; font-weight: 500;">({attack_count/len(data)*100:.1f}%)</span></div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m4:
                        avg_confidence = np.mean(np.max(binary_proba, axis=1))
                        st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{avg_confidence:.1%}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Attack details
                    if attack_count > 0:
                        st.warning(f"⚠️ Warning: {attack_count} potential DDoS attacks detected!")
                        
                        if len(attack_pred) > 0:
                            col_chart1, col_chart2 = st.columns(2)
                            
                            with col_chart1:
                                st.markdown("### Attack Type Breakdown")
                                attack_df = pd.DataFrame({'Attack_Type': attack_pred})
                                attack_counts = attack_df['Attack_Type'].value_counts()
                                
                                for attack_type, count in attack_counts.items():
                                    percentage = (count / attack_count) * 100
                                    st.markdown(f"""
                                        <div style="background: #f3f4f6; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                                            <strong>{attack_type}</strong>
                                            <span style="float: right; color: #065f46;">{count} ({percentage:.1f}%)</span>
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            with col_chart2:
                                st.markdown("### Attack Distribution")
                                
                                # Create vibrant color palette
                                color_palette = [
                                    '#ef4444',  # Red
                                    '#f97316',  # Orange
                                    '#8b5cf6',  # Purple
                                    '#3b82f6',  # Blue
                                    '#06b6d4',  # Cyan
                                    '#ec4899',  # Pink
                                    '#f59e0b',  # Amber
                                    '#10b981',  # Green
                                    '#6366f1',  # Indigo
                                    '#14b8a6'   # Teal
                                ]
                                
                                # Ensure we have enough colors
                                num_attacks = len(attack_counts)
                                colors = color_palette[:num_attacks] if num_attacks <= len(color_palette) else color_palette * (num_attacks // len(color_palette) + 1)
                                colors = colors[:num_attacks]
                                
                                fig = go.Figure(data=[go.Pie(
                                    labels=attack_counts.index.tolist(),
                                    values=attack_counts.values.tolist(),
                                    hole=0.4,
                                    marker=dict(
                                        colors=colors,
                                        line=dict(color='#ffffff', width=3)
                                    ),
                                    textposition='inside',
                                    textinfo='percent',
                                    textfont=dict(size=14, color='white', family='Alan Sans'),
                                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                                )])
                                
                                fig.update_layout(
                                    height=400,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="v",
                                        yanchor="middle",
                                        y=0.5,
                                        xanchor="left",
                                        x=1.02,
                                        font=dict(size=12)
                                    ),
                                    margin=dict(l=20, r=120, t=20, b=20),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key=f"pie_chart_{time.time()}")
                    else:
                        st.success("✓ All clear! No attacks detected in the network traffic.")
                    
                    # Download results
                    st.markdown("### Download Results")
                    results_df = data.copy()
                    results_df['Prediction'] = ['Attack' if p == 1 else 'Benign' for p in binary_pred]
                    results_df['Confidence'] = np.max(binary_proba, axis=1)
                    
                    attack_type_column = [''] * len(data)
                    if len(attack_pred) > 0:
                        attack_idx_list = np.where(attack_indices)[0]
                        for i, attack_type in enumerate(attack_pred):
                            attack_type_column[attack_idx_list[i]] = attack_type
                    
                    results_df['Attack_Type'] = attack_type_column
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Detailed Analysis Report",
                        data=csv,
                        file_name="ddos_analysis_report.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()