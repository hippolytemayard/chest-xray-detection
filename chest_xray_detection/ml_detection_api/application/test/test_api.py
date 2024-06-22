import requests


def send_image_to_api(url: str, image_path: str, params: dict = None, headers: dict = None) -> dict:
    """
    Send an image to an API.

    Args:
    - url (str): The API endpoint URL.
    - image_path (str): The path to the image file.
    - params (dict): Optional query parameters for the request.
    - headers (dict): Optional headers for the request.

    Returns:
    - dict: The JSON response from the API.
    """
    try:
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, files=files)  # , params=params)  # , headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()  # Parse JSON response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":

    image_path = "/home/ubuntu/data/images/00010366_000.png"
    api_url = "http://127.0.0.1:8000/api/pathology-detection"
    query_params = {"param1": "value1", "param2": "value2"}
    # custom_headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN"}

    response_data = send_image_to_api(api_url, image_path, params=query_params)

    if response_data:
        print(response_data)
