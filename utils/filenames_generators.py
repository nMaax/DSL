import time

def generate_model_filename(model, prefix="model"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Current timestamp
    model_name = type(model).__name__  # Model type
        
    params = str(model.get_params()).replace(" ", "").replace("\n", "")[:50]  # First 50 characters of the parameters

    filename = f"{prefix}-{timestamp}-{model_name}"
    
    return filename

def generate_dataset_filename():
    filename = 'data\\' + 'development-' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
    return filename