import cv2
from inference_sdk import InferenceHTTPClient
import pytesseract
import os
import base64
from fastapi import FastAPI

app = FastAPI()
@app.post("/decide-road")
def decide_road(id):
    '''
    Gets id value and returns the right pash for that value
    :param id: string
    :return: list of base64_string
    '''

    roads_path = "itc2024/"
    if id == "89839":
        roads_path += "m2"
    elif id == "89768":
        roads_path += "m3"
    errors = check_road(roads_path)
    return {"error_roads": errors}

def get_pothole_prediction(image_path):
    '''
    Pothole detection based on a huggin face model. Sends request to roboflow and gets predictions for placing plothols.
    Options: builds boxes on the image around plotholes
    :param image_path: string
    :return: True -> potholes found
            False -> the road is good
    '''
    # Set up the Inference HTTP Client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="iFZGaJVqJ8m7sjJioKym"
    )
    from PIL import Image

    img = Image.open(image_path)
    img = img.quantize(colors=64)  # Уменьшение до 64 цветов
    processed_image_path = 'reduced_colors_image.png'
    img.save(processed_image_path)

    # Perform inference
    result = CLIENT.infer(processed_image_path, model_id="pothole-jujbl/1")

    # Load the image
    image = cv2.imread(image_path)

    if len(result['predictions']) != 0:
        # Iterate over the predictions
        for prediction in result["predictions"]:
            # Get the bounding box coordinates
            x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]

            x, y, w, h = int(x), int(y), int(w), int(h)
            # Draw the bounding box on the image
            cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w, y + h), (0, 255, 0), 2)

            # Get the class label and confidence score
            class_label = prediction["class"]
            confidence = prediction["confidence"]

            # Put the class label and confidence score on the image
            label = f"{class_label}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        np_array = image.copy()
        base64_string = base64.b64encode(np_array).decode('utf-8')
        return base64_string

    return None


def get_text_extracted(image_path):
    '''
    Extracts text from the image using tesseract
    :param image_path: string
    :return: text: string
    '''

    # Load the image
    image = cv2.imread(image_path)

    # Perform OCR with Russian language
    text = pytesseract.image_to_string(image, lang='rus')
    return text


def check_road(roads):
    '''
    Checks photos of the road and decide, whether this road needs a repair.
    Return photos where were detected potholes encoded in base64
    :param roads: path to folder containing road photos
    :return: list of base64 encoded images
                [] -> no problems
                if len(list) > 0 -> [base64_image1, base64_image2, ...]
    '''
    error_roads = []

    # Iterate over each file in the folder
    for filename in os.listdir(roads):
        filepath = os.path.join(roads, filename)

        # Check if the current item is a file (not a directory)
        if os.path.isfile(filepath):
            try:
                # Perform pothole detection (assuming you have a function called get_pothole_prediction)
                error = get_pothole_prediction(filepath)
                if error is not None:
                    error_roads.append(error)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    return error_roads


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)