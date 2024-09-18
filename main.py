import cv2
from inference_sdk import InferenceHTTPClient
import pytesseract
from pathlib import Path
import os

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
    img = img.quantize(colors=128)  # Уменьшение до 256 цветов
    processed_image_path = 'reduced_colors_image.png'
    img.save(processed_image_path)

    # Perform inference
    result = CLIENT.infer(processed_image_path, model_id="pothole-jujbl/1")
    os.remove(processed_image_path)
    # Load the image
    # image = cv2.imread(image_path)
    # cv2.imshow("Inference Result", image)

    # Iterate over the predictions
    # for prediction in result["predictions"]:
    #     # Get the bounding box coordinates
    #     x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
    #
    #     x, y, w, h = int(x), int(y), int(w), int(h)
    #     # Draw the bounding box on the image
    #     cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w, y + h), (0, 255, 0), 2)
    #
    #     # Get the class label and confidence score
    #     class_label = prediction["class"]
    #     confidence = prediction["confidence"]
    #
    #     # Put the class label and confidence score on the image
    #     label = f"{class_label}: {confidence:.2f}"
    #     cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image with bounding boxes and labels
    # print(result)
    # cv2.imshow("Inference Result", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if len(result['predictions']) == 0:
        return False
    return True


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
    Checks photos of the road and decide, wheather this road needs a remont
    :param roads: list of strings
    :return: True -> bad road
            False -> good road
    '''
    # Specify the path to the "roads" folder
    roads_folder = Path(roads)
    roads_links = []
    # Iterate over each file in the folder
    for file_path in roads_folder.iterdir():
        # Check if the current item is a file (not a directory)
        if file_path.is_file():
            # Process the file
            roads_links.append(str(file_path))
    return any(get_pothole_prediction(road) for road in roads_links)

# print(get_pothole_prediction(image_path = "itc2024/road/3.png"))

# print(get_text_extracted(image_path="itc2024/assets/img_6.png"))

print(check_road("itc2024/road"))