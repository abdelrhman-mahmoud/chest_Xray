import gdown
import os
import zipfile
import shutil

def download_extract_model():
    if os.path.exists("models"):
        print("Models folder already exists. Skipping download.")
        return
    
    os.makedirs("models", exist_ok=True)
    
    file_id = "1n8QwV6FhczWd8CufxUJ83tpUmHSNPnbT"
    url = f"https://drive.google.com/uc?id={file_id}"
    output_zip = "models.zip"
    
    print("Downloading model...")
    gdown.download(url, output_zip, quiet=False)
    
    print("Extracting model...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall("models")
    
    print("Cleaning up...")
    os.remove(output_zip)
    print("Model downloaded and extracted successfully.")

if __name__ == "__main__":
    download_extract_model()