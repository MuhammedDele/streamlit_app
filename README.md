
# Streamlit Prediction Application

This repository contains a Dockerized Streamlit application designed for predictive modeling. Users can upload a dataset, train a Random Forest model, visualize data insights, and download the trained model.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed on your system:

1. **Docker**: Download and install Docker Desktop from [Docker's official website](https://www.docker.com/products/docker-desktop).
2. **A Dataset**: Prepare a CSV file for upload. Ensure the last column in your dataset is the target variable to be predicted, and the other columns are features.

---

### Pull the Docker Image

Retrieve the pre-built Docker image from Docker Hub by running:
```sh
docker pull <your-username>/streamlit-app:latest
```

---

## Running the Streamlit Application

### Start the Docker Container

Run the Docker container with the following command:
```sh
docker run -p 8501:8501 <your-username>/streamlit-app:latest
```

Open your web browser and go to [http://localhost:8501](http://localhost:8501) to access the application.

---

## Using the Streamlit Application

### Step 1: Upload a Dataset

- Click the **"Choose a CSV file"** button in the application interface.
- Select a CSV file from your local system. Ensure:
  - The last column is the target variable (e.g., the column to be predicted).
  - The file includes headers for each column.

### Step 2: View Data Insights

- After uploading the dataset, the application displays:
  - A preview of the dataset.
  - Summary statistics, including dataset shape and missing values.

### Step 3: Train the Model

- The application automatically trains a Random Forest model on the dataset.
- Metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score are displayed to evaluate the model's performance.

### Step 4: Visualize Data

The application provides:
- **Correlation Heatmap**: Shows relationships between numerical features.
- **Feature Distributions**: Displays histograms of numerical features.
- **Prediction Analysis**:
  - Predicted vs. Actual scatter plot.
  - Residuals plot.
- **Feature Importance**: Highlights the top features influencing the model's predictions.

### Step 5: Download the Model

- After training, you can download the trained model pipeline as a `.pkl` file by clicking **"Download trained model pipeline"**.

---

## Example Dataset

You can test the application with any dataset where:
- The last column is the target variable to predict.
- Features are in numeric or categorical format.

For testing, a dummy dataset is provided in this repository (`dummy_dataset.csv`).

---

## Troubleshooting

### Docker Not Running

Ensure Docker is installed and running. Open Docker Desktop if needed and retry.

### Port Conflicts

If port `8501` is in use, map the container to a different port:
```sh
docker run -p <host_port>:8501 <your-username>/streamlit-app:latest
```
Replace `<host_port>` with an available port number, e.g., `8502`.

---

### Feedback & Contributions

If you encounter any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

Happy predicting! 🚀
