import pickle
import numpy as np

# Debugging function to check model loading
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            
            # Additional checks to verify model type
            if hasattr(model, 'predict'):
                print("Model loaded successfully!")
                return model
            else:
                print("Loaded object does not appear to be a valid machine learning model.")
                return None
    
    except FileNotFoundError:
        print(f"Error: Model file not found at {file_path}")
        return None
    except pickle.UnpicklingError:
        print("Error: Unable to unpickle the model. The file might be corrupted.")
        return None
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        return None

# Load the model
model_path = 'fertilizer_type_model.pkl'
model = load_model(model_path)

# Example input data
input_data = [
    [0,150,80,150,7.0,1400,30,1]  # Example input features
]
input_data_reshaped = np.array(input_data)

# Prediction (with error handling)
if model is not None:
    try:
        prediction = model.predict(input_data_reshaped)
        print("Prediction:", prediction)
    except Exception as e:
        print(f"Prediction error: {e}")
        print("Model type:", type(model))
else:
    print("Cannot make prediction. Model not loaded correctly.")


















0,150,80,150,7.0,1400,30,1