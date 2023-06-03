import requests
import os
import shutil
from zipfile import ZipFile
from tqdm import tqdm


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
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(block_size):
            progress_bar.update(len(chunk))
            if chunk:
                f.write(chunk)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during the download and unzip step")


def main():
    print("Build dataset - Start procedure")
    if not os.path.exists('./dataset/development.csv') or not os.path.exists('./dataset/evaluation.csv'):
        download_file_from_google_drive('1QTyU4vNW3WrMIkIQwUtl7cEmm0nVi-IL', './dataset.zip')

        with ZipFile('./dataset.zip', 'r') as zip:
            zip.extractall('./dataset')

    os.remove('./dataset.zip')
    shutil.rmtree('./dataset/__MACOSX')
    print("Build dataset - End procedure")


if __name__ == "__main__":
    main()
