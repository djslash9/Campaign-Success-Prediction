import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import base64
import matplotlib.pyplot as plt

# Page configuration
# Hide the footer and fork button

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Campaign Success Predictor", layout="wide")
st.title("🎯 Campaign Success Prediction")

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'test_data_pred' not in st.session_state:
    st.session_state.test_data_pred = None

# Knowledge base for explanations
FAILURE_REASONS = {
    'ROI': "Low ROI (< 2.0) reduces success likelihood. Try optimizing ad spend or targeting higher-value customers.",
    'Engagement_Score': "Scores below 5 indicate poor audience interaction. Consider more engaging content formats.",
    'Conversion_Rate': "Rates under 8% often lead to failure. Test different CTAs or landing page designs.",
    'Clicks': "Insufficient clicks suggest weak ad appeal. A/B test different creatives and messaging.",
    'Impressions': "Low impressions mean limited reach. Increase budget or improve targeting parameters.",
    'Channel_Used': "Some channels perform better than others for specific demographics. Review channel performance metrics."
}

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Upload & Predict", "Campaign Diagnostics"])
st.sidebar.markdown("---")
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV", type=["csv"])

def show_diagnostics():
    st.header("📊 Campaign Failure Analysis")
    
    if st.session_state.test_data_pred is not None:
        failed_campaigns = st.session_state.test_data_pred[
            st.session_state.test_data_pred['Prediction'] == 0
        ]
        
        if not failed_campaigns.empty:
            st.subheader("Select a Campaign to Analyze")
            
            # Create campaign selection dropdown with IDs
            campaign_options = failed_campaigns['Campaign_ID'].unique()[:50]  # Show first 50 for performance
            selected_campaign = st.selectbox(
                "Campaign ID:",
                options=campaign_options,
                index=0
            )
            
            # Get the selected campaign data
            campaign_data = failed_campaigns[failed_campaigns['Campaign_ID'] == selected_campaign].iloc[0]
            
            # Display campaign metrics in columns
            st.subheader(f"🔍 Analysis for Campaign: {selected_campaign}")
            
            # Metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ROI", f"{campaign_data['ROI']:.2f}")
                st.metric("Engagement", f"{campaign_data['Engagement_Score']:.1f}")
            with col2:
                st.metric("Conversion Rate", f"{campaign_data['Conversion_Rate']:.1%}")
                st.metric("Clicks", f"{int(campaign_data['Clicks']):,}")
            with col3:
                st.metric("Impressions", f"{int(campaign_data['Impressions']):,}")
                st.metric("Channel", campaign_data['Channel_Used'])
            
            # Generate and display improvement suggestions
            st.subheader("🛠️ Improvement Suggestions")
            suggestions = []
            
            if campaign_data['ROI'] < 2: 
                suggestions.append(FAILURE_REASONS['ROI'])
            if campaign_data['Engagement_Score'] < 5: 
                suggestions.append(FAILURE_REASONS['Engagement_Score'])
            if campaign_data['Conversion_Rate'] < 0.08: 
                suggestions.append(FAILURE_REASONS['Conversion_Rate'])
            if campaign_data['Clicks'] < 100: 
                suggestions.append(FAILURE_REASONS['Clicks'])
            if campaign_data['Impressions'] < 1000: 
                suggestions.append(FAILURE_REASONS['Impressions'])
            
            if suggestions:
                for suggestion in suggestions:
                    st.markdown(f"📌 {suggestion}")
            else:
                st.info("No clear failure patterns detected in primary metrics")
            
            # Show probability
            st.metric("Success Probability", f"{campaign_data['Success_Probability']:.1%}",
                     help="Model's confidence in campaign success")
            
            # Display full campaign data in |Column|Value| table
            st.subheader("📋 Campaign Details")
            
            # Prepare data for display
            display_data = pd.DataFrame({
                'Column': campaign_data.index,
                'Value': campaign_data.values
            }).reset_index(drop=True)
            
            # Format specific columns for better readability
            def format_value(val):
                if isinstance(val, float):
                    if 0 < val < 1:  # Likely a percentage
                        return f"{val:.1%}"
                    return f"{val:.2f}"
                elif isinstance(val, (int, np.integer)):
                    return f"{val:,}"
                return str(val)
            
            display_data['Value'] = display_data['Value'].apply(format_value)
            
            # Show the table with custom styling
            st.table(display_data.style
                    .set_properties(**{'text-align': 'left'})
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('background-color', '#f0f2f6'), 
                                 ('font-weight', 'bold')]
                    }]))
            
        else:
            st.success("🎉 No failed campaigns in this dataset!")
    else:
        st.warning("⚠️ Please make predictions first from the Upload & Predict page")
        
# Main function to handle the app logic
def main():
    if menu == "Upload & Predict":
        st.header("📤 Upload & Predict")
        
        if uploaded_file is not None:
            # Load and show data first
            test_data = pd.read_csv(uploaded_file)
            st.subheader("📂 Uploaded Data Preview")
            st.dataframe(test_data.head())
            
            # Add prediction button
            if st.button("🔮 Predict Success", type="primary", help="Run the prediction model on your data"):
                with st.spinner('🧠 Making predictions...'):
                    test_data_pred = test_data.copy()
                    
                    # Load models
                    loaded_scaler = joblib.load('scaler.pkl')
                    logreg = joblib.load('logreg_model.pkl')
                    
                    # Data validation and preprocessing
                    required_columns = ['ROI', 'Engagement_Score', 'Conversion_Rate', 
                                      'Clicks', 'Impressions', 'Channel_Used']
                    
                    test_data_processed = test_data[required_columns].copy()
                    channel_map = {'Facebook': 0, 'Instagram': 1, 'Pinterest': 2, 'Twitter': 3}
                    test_data_processed['Channel_Used'] = test_data_processed['Channel_Used'].map(channel_map)
                    test_data_processed = test_data_processed.fillna(0)
                    test_data_scaled = loaded_scaler.transform(test_data_processed)
                    
                    # Make predictions
                    predictions = logreg.predict(test_data_scaled)
                    probabilities = logreg.predict_proba(test_data_scaled)[:, 1]
                    
                    # Add results
                    test_data_pred['Success_Probability'] = probabilities
                    test_data_pred['Prediction'] = predictions
                    test_data_pred['Prediction_Label'] = test_data_pred['Prediction'].map({0: 'Failed', 1: 'Success'})
                    
                    # Store in session state
                    st.session_state.test_data_pred = test_data_pred
                    st.session_state.predictions_made = True
                    
                    # Show results
                    st.success("✅ Predictions completed!")
                    success_rate = test_data_pred['Prediction'].mean()
                    st.metric("Overall Success Rate", f"{success_rate:.1%}")
                    
                    st.subheader("📊 Prediction Results")
                    st.dataframe(test_data_pred.head())
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    with col1:
                        st.bar_chart(test_data_pred['Prediction_Label'].value_counts())
                    
                    with col2:
                        st.write("📈 Success Probability Distribution")
                        fig, ax = plt.subplots()
                        ax.hist(probabilities, bins=20, color='skyblue', edgecolor='black')
                        ax.set_xlabel('Success Probability')
                        ax.set_ylabel('Count')
                        st.pyplot(fig)
                    
                    # Download button
                    csv = test_data_pred.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="campaign_predictions.csv">💾 Download Full Results</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Add analysis suggestion
                    if (test_data_pred['Prediction'] == 0).any():
                        st.markdown("---")
                        st.success("🔍 Switch to the 'Campaign Diagnostics' tab in the sidebar to analyze failed campaigns")
        else:
            st.info("📤 Please upload a CSV file to get predictions")
    
    elif menu == "Campaign Diagnostics":
        show_diagnostics()

if __name__ == "__main__":
    main()