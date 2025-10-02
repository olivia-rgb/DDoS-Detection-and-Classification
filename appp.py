import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Configure page
st.set_page_config(page_title="DDoS Detection System", page_icon="🛡️", layout="wide")

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

def predict_traffic(data, models):
    """Predict traffic using XGBoost with two-stage classification"""
    try:
        expected_features = models['feature_names']
        X = data.drop(columns=["label", "activity"], errors='ignore')
        
        # Check for missing features
        missing_features = set(expected_features) - set(X.columns)
        if missing_features:
            st.error(f"Missing required features: {list(missing_features)[:10]}...")
            return None, None, None, None
        
        # Select features in correct order
        X = X[expected_features]
        X_scaled = models['scaler'].transform(X)
        
        # Stage 1: Binary prediction (Benign vs Attack/Suspicious)
        binary_pred = models['xgb_binary'].predict(X_scaled)
        binary_proba = models['xgb_binary'].predict_proba(X_scaled)
        
        # Stage 2: Attack type prediction for detected attacks
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
    st.title("🛡️ DDoS Detection System")
    st.markdown("Upload network traffic data to detect and classify DDoS attacks using XGBoost")
    
    models = load_models()
    if models is None:
        st.stop()
    
    # Show dataset info
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Model Information")
        st.info(f"**Features Required:** {len(models['feature_names'])}")
        st.info(f"**Attack Types:** {len(models['label_encoder'].classes_)}")
        
        with st.expander("Attack Types Detected"):
            for attack_type in models['label_encoder'].classes_:
                st.write(f"• {attack_type}")
        
        with st.expander("View All Required Features"):
            st.text('\n'.join(models['feature_names']))
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file with network traffic data",
            type="csv",
            help="File should contain the exact same features used for training"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
                
                with st.expander("View Sample Data"):
                    st.dataframe(data.head())
                
                if st.button("🔍 Analyze Traffic", type="primary"):
                    with st.spinner("Analyzing traffic..."):
                        binary_pred, binary_proba, attack_pred, attack_indices = predict_traffic(data, models)
                        
                        if binary_pred is not None:
                            benign_count = sum(binary_pred == 0)
                            attack_count = sum(binary_pred == 1)
                            
                            st.subheader("📊 Analysis Results")
                            
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Total Samples", len(data))
                            with col_m2:
                                st.metric("Benign Traffic", benign_count, 
                                         delta=f"{benign_count/len(data)*100:.1f}%")
                            with col_m3:
                                st.metric("Attack Traffic", attack_count, 
                                         delta=f"{attack_count/len(data)*100:.1f}%")
                            
                            if attack_count > 0:
                                st.warning(f"⚠️ {attack_count} potential DDoS attacks detected!")
                                
                                if len(attack_pred) > 0:
                                    st.subheader("🎯 Attack Type Distribution")
                                    attack_df = pd.DataFrame({'Attack_Type': attack_pred})
                                    attack_counts = attack_df['Attack_Type'].value_counts()
                                    
                                    # Display attack counts
                                    for attack_type, count in attack_counts.items():
                                        st.write(f"• **{attack_type}**: {count} samples")
                                    
                                    # Pie chart
                                    fig = px.pie(
                                        values=attack_counts.values,
                                        names=attack_counts.index,
                                        title="Attack Type Distribution"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.success("✅ No attacks detected in the uploaded data")
                            
                            # Confidence scores
                            avg_confidence = np.mean(np.max(binary_proba, axis=1))
                            st.metric("Average Model Confidence", f"{avg_confidence:.3f}")
                            
                            # Download results
                            results_df = data.copy()
                            results_df['Prediction'] = ['Attack' if p == 1 else 'Benign' for p in binary_pred]
                            results_df['Confidence'] = np.max(binary_proba, axis=1)
                            
                            # Add attack types where applicable
                            attack_type_column = [''] * len(data)
                            if len(attack_pred) > 0:
                                attack_idx_list = np.where(attack_indices)[0]
                                for i, attack_type in enumerate(attack_pred):
                                    attack_type_column[attack_idx_list[i]] = attack_type
                            
                            results_df['Attack_Type'] = attack_type_column
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Detailed Results",
                                data=csv,
                                file_name="ddos_analysis_results.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error loading file: {e}")

if __name__ == "__main__":
    main()