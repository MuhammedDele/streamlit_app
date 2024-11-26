import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def get_column_types(data):
    """Identify numerical and categorical columns"""
    numerical_cols = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in data.columns if data[cname].dtype == 'object']
    return numerical_cols, categorical_cols

def create_pipeline(numerical_cols, categorical_cols):
    """Create preprocessing and model pipeline"""
    numerical_transformer = SimpleImputer(strategy='constant')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=0))
    ])
    
    return model_pipeline

def plot_correlation_heatmap(data, numerical_cols):
    """Create correlation heatmap for numerical columns"""
    correlation = data[numerical_cols].corr()
    fig = px.imshow(correlation,
                    labels=dict(color="Correlation"),
                    title="Feature Correlation Heatmap")
    return fig

def plot_feature_distributions(data, numerical_cols):
    """Create distribution plots for numerical features"""
    fig = go.Figure()
    for col in numerical_cols:
        fig.add_trace(go.Histogram(x=data[col], name=col, opacity=0.7))
    fig.update_layout(
        title="Numerical Feature Distributions",
        xaxis_title="Value",
        yaxis_title="Count",
        barmode='overlay'
    )
    return fig

def plot_prediction_vs_actual(y_valid, predictions, target_column):
    """Create scatter plot of predicted vs actual values"""
    fig = px.scatter(x=y_valid, y=predictions,
                    labels={'x': f'Actual {target_column}', 'y': f'Predicted {target_column}'},
                    title=f'Predicted vs Actual {target_column}')
    fig.add_trace(
        go.Scatter(x=[y_valid.min(), y_valid.max()],
                  y=[y_valid.min(), y_valid.max()],
                  mode='lines',
                  name='Perfect Prediction',
                  line=dict(color='red', dash='dash'))
    )
    return fig

def plot_residuals(y_valid, predictions, target_column):
    """Create residual plot"""
    residuals = predictions - y_valid
    fig = px.scatter(x=predictions, y=residuals,
                    labels={'x': f'Predicted {target_column}', 'y': 'Residuals'},
                    title=f'Residual Plot ({target_column})')
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig

def train_model(data):
    """Train the model and return pipeline, predictions, and metrics"""
    target_column = data.columns[-1]  # Use the last column as the target
    y = data[target_column]
    X = data.drop([target_column], axis=1)
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )
    
    numerical_cols, categorical_cols = get_column_types(X_train)
    pipeline = create_pipeline(numerical_cols, categorical_cols)
    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_valid)
    
    metrics = {
        'mae': mean_absolute_error(y_valid, predictions),
        'rmse': np.sqrt(mean_squared_error(y_valid, predictions)),
        'r2': r2_score(y_valid, predictions)
    }
    
    return pipeline, predictions, y_valid, metrics, numerical_cols, target_column

# Streamlit app
st.title("Flexible Prediction Application")
st.write("""
Upload a dataset for regression. The last column in the dataset will be considered 
the target variable, and the remaining columns will be used as features.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Data Overview Section
    st.header("Data Overview")
    st.write("Data Preview:")
    st.write(data.head())
    
    st.write("Dataset Shape:", data.shape)
    st.write("Missing Values:", data.isnull().sum().sum())
    
    # Train model and get results
    with st.spinner('Training model...'):
        pipeline, predictions, y_valid, metrics, numerical_cols, target_column = train_model(data)
    
    # Model Performance Metrics
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
    with col2:
        st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
    with col3:
        st.metric("R² Score", f"{metrics['r2']:.3f}")
    
    st.write(f"Target column used for prediction: **{target_column}**")
    
    # Data Visualization Section
    st.header("Data Visualization")
    
    # Correlation Heatmap
    st.subheader("Feature Correlations")
    correlation_fig = plot_correlation_heatmap(data, numerical_cols)
    st.plotly_chart(correlation_fig)
    
    # Feature Distributions
    st.subheader("Feature Distributions")
    dist_fig = plot_feature_distributions(data, numerical_cols)
    st.plotly_chart(dist_fig)
    
    # Model Predictions Visualization
    st.header("Model Predictions Analysis")
    
    # Predicted vs Actual
    st.subheader(f"Predicted vs Actual {target_column}")
    pred_vs_actual_fig = plot_prediction_vs_actual(y_valid, predictions, target_column)
    st.plotly_chart(pred_vs_actual_fig)
    
    # Residuals Plot
    st.subheader("Residuals Analysis")
    residuals_fig = plot_residuals(y_valid, predictions, target_column)
    st.plotly_chart(residuals_fig)
    
    # Feature Importance
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        st.header("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': pipeline.named_steps['preprocessor'].get_feature_names_out(),
            'importance': pipeline.named_steps['model'].feature_importances_
        })
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False
        ).head(10)
        
        fig = px.bar(feature_importance, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title='Top 10 Most Important Features')
        st.plotly_chart(fig)
    
    # Model Download Section
    st.header("Download Trained Model")
    model_filename = "random_forest_pipeline.pkl"
    joblib.dump(pipeline, model_filename)
    
    with open(model_filename, "rb") as file:
        btn = st.download_button(
            label="Download trained model pipeline",
            data=file,
            file_name=model_filename,
            mime="application/octet-stream"
        )
