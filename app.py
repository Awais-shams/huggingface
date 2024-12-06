import gradio as gr
import mlflow.pyfunc
import pandas as pd

# Load feature names from the uploaded features file 
features_file = "./features.txt"  # Update the path if different
with open(features_file, "r") as f:
    features = [line.strip().split()[1] for line in f.readlines()]

# Load the MLflow model
model = mlflow.pyfunc.load_model('./model')  # Path to the downloaded model directory

# Define the prediction function
def predict(data):
    # Convert input data to a pandas DataFrame with the feature names
    input_df = pd.DataFrame(data, columns=features)
    # Predict using the MLflow model
    predictions = model.predict(input_df)
    return predictions.tolist()

# Create Gradio interface
# Define the input interface for Gradio using the feature list
input_interface = gr.Dataframe(
    headers=features,  # Use the full feature list (561 features)
    datatype="number",
    row_count=1,  # Allow one instance of input
)

# Define the output interface
output_interface = gr.Dataframe(
    headers=["Prediction"],  # Adjust as needed for your prediction format
    datatype="number"
)

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=input_interface,
    outputs=output_interface,
    title="Random Forest Better Accuracy",
    description="Predict human activity (e.g., walking, standing, etc.) based on smartphone sensor data. Provide all 561 feature values as input.",
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
