import requests
from typing import Optional


def send_image_to_api(url: str, image_path: str) -> Optional[dict]:
    """
    Sends an image file to an API endpoint using HTTP POST request with multipart/form-data.

    Args:
        url (str): The URL of the API endpoint.
        image_path (str): The file path of the image to send.

    Returns:
        Optional[dict]: A dictionary containing the JSON response from the API if successful, None otherwise.
    """
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, files=files)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()  # Parse JSON response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while sending image to API: {e}")
        return None


if __name__ == "__main__":
    image_path = "chest_xray_detection/ml_detection_api/application/test/data/00000032_037.png"
    api_url = "http://127.0.0.1:8000/api/pathology-detection"

    response_data = send_image_to_api(api_url, image_path)

    if response_data:
        print(response_data)
