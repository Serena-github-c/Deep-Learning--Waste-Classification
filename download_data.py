# script to donwload the dataset from kaggle
import os
import zipfile

def download_dataset():
    dataset_path = "waste-segregation-dataset"
    if not os.path.exists(dataset_path):
        print("Dataset not found. Downloading from Kaggle...")

        # Ensure the user has their Kaggle API key
        if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            raise Exception("Kaggle API key not found! Please follow setup instructions in the README.")

        # Download the dataset
        os.system("kaggle datasets download -d aashidutt3/waste-segregation-image-dataset -p .")

        # Extract the dataset
        zip_path = "waste-segregation-image-dataset.zip"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)
        
        # Clean up
        os.remove(zip_path)

        print("Dataset downloaded and extracted successfully!")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_dataset()
