import requests
import os
import shutil
from zipfile import ZipFile

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def main():

    print("Build dataset - Start procedure")
    if not os.path.exists('./dataset/development.csv'):
        download_file_from_google_drive('1QTyU4vNW3WrMIkIQwUtl7cEmm0nVi-IL', './dataset.zip')
            
        with ZipFile('./dataset.zip', 'r') as zip:
            zip.extractall('./dataset')

    os.remove('./dataset.zip', )
    shutil.rmtree('./dataset/__MACOSX')



if __name__ == "__main__":
    main()