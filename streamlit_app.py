import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def get_column_types(data):
    """Identify numerical and categorical columns"""
    numerical_cols = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in data.columns if data[cname].dtype == 'object']
    return numerical_cols, categorical_cols

def create_pipeline(numerical_cols, categorical_cols):
    """Create preprocessing and model pipeline"""
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Bundle preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Create the full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=0))
    ])
    
    return model_pipeline

def train_model(data, target_column):
    """Train the model and return pipeline, predictions, and score"""
    # Separate target from predictors
    y = data[target_column]
    X = data.drop([target_column], axis=1)
    
    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=0
    )
    
    # Get column types
    numerical_cols, categorical_cols = get_column_types(X_train)
    
    # Create and train pipeline
    pipeline = create_pipeline(numerical_cols, categorical_cols)
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    predictions = pipeline.predict(X_valid)
    
    # Calculate score
    mae_score = mean_absolute_error(y_valid, predictions)
    
    return pipeline, predictions, mae_score

# Streamlit app
st.title("Housing Price Predictor")
st.write("""
This application trains a Random Forest model to predict housing prices. 
Upload your dataset with the target column named 'Price'.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load and display data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())
    
    # Train model
    with st.spinner('Training model...'):
        pipeline, predictions, mae_score = train_model(data, 'Price')
    
    # Display results
    st.write(f"Model Mean Absolute Error: ${mae_score:,.2f}")
    
    # Save the model
    model_filename = "random_forest_pipeline.pkl"
    joblib.dump(pipeline, model_filename)
    
    # Provide download button
    with open(model_filename, "rb") as file:
        btn = st.download_button(
            label="Download model pipeline",
            data=file,
            file_name=model_filename,
            mime="application/octet-stream"
        )
    
    # Display feature importance if available
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        st.write("\nFeature Importance:")
        feature_importance = pd.DataFrame({
            'feature': pipeline.named_steps['preprocessor']\
                .get_feature_names_out(),
            'importance': pipeline.named_steps['model'].feature_importances_
        })
        feature_importance = feature_importance.sort_values(
            'importance', ascending=False
        ).head(10)
        st.bar_chart(data=feature_importance.set_index('feature'))