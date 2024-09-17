import cv2
from inference_sdk import InferenceHTTPClient
import pytesseract


def get_pothole_prediction(image_path):
    # Set up the Inference HTTP Client
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="iFZGaJVqJ8m7sjJioKym"
    )

    # Perform inference
    result = CLIENT.infer(image_path, model_id="pothole-jujbl/1")

    # Load the image
    image = cv2.imread(image_path)
    cv2.imshow("Inference Result", image)

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

    # Display the image with bounding boxes and labels
    print(result)
    cv2.imshow("Inference Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_text_extracted(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Perform OCR with Russian language
    text = pytesseract.image_to_string(image, lang='rus')
    print(text)

# get_pothole_prediction(image_path = "itc2024/assets/img_5.png")
# get_text_extracted(image_path="assets/passport_image.png")
