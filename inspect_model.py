import torch

# --- IMPORTANT: Change this to the actual path of your .pt file ---
model_path = r'.\models\best_ckpt_2cls.pt'

print(f"Inspecting model file: {model_path}")

try:
    # Load the checkpoint file. It's usually a dictionary.
    # map_location=torch.device('cpu') ensures it loads even without a GPU.
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # --- The most likely place to find the names is in a 'names' key ---
    # Sometimes it's inside a 'model' object within the checkpoint.
    
    class_names = None
    
    # First, check for a top-level 'names' key
    if 'names' in checkpoint:
        class_names = checkpoint['names']
        print("Found class names in the 'names' key!")

    # If not found, check inside a 'model' object
    elif 'model' in checkpoint and hasattr(checkpoint['model'], 'names'):
        class_names = checkpoint['model'].names
        print("Found class names in the 'model.names' attribute!")
        
    else:
        print("\nCould not find a 'names' key directly.")
        print("Here are all the available keys in the file:")
        # If it's a dictionary, print its keys
        if isinstance(checkpoint, dict):
            print(list(checkpoint.keys()))
        else:
            print("The file is not a standard dictionary checkpoint.")

    if class_names:
        print("\n--- CLASS NAMES ---")
        for i, name in enumerate(class_names):
            print(f"Class {i}: {name}")
        print("--------------------")

except Exception as e:
    print(f"\nAn error occurred while trying to load the model: {e}")