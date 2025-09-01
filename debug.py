import torch

def print_pt_sample(file_path):
    try:
        # Load the .pt file; using map_location to ensure compatibility with CPU
        data = torch.load(file_path, map_location=torch.device('cpu'))
        
        # If the file is a dictionary (as many checkpoints are)
        if isinstance(data, dict):
            print("Keys in the .pt file:")
            for key, value in data.items():
                print(f"\nKey: {key}")
                if isinstance(value, torch.Tensor):
                    # Print tensor shape and a small sample (first 5 elements)
                    print("Tensor shape:", value.shape)
                    flat_val = value.flatten()
                    sample = flat_val[:5] if flat_val.numel() >= 5 else flat_val
                    print("Tensor sample:", sample)
                else:
                    # Print the value directly if not a tensor
                    print("Value:", value)
        else:
            # If not a dictionary, simply print the data
            print("Data loaded from .pt file:")
            print(data)
    except Exception as e:
        print("Error loading the .pt file:", e)

# Example usage:
if __name__ == "__main__":
    print_pt_sample("tensor_data/Anderssen.pt")
