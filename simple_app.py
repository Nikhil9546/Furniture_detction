import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.pt")

# Define the Streamlit app


def main():
    st.title("Image Upload and Processing App")
    st.write("Upload an image and see the processed result below:")

    # Create a file uploader widget
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"])

    # Check if an image has been uploaded
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image",
                 use_column_width=True)

        # Process the uploaded image using the YOLO model
        if st.button("Process"):
            img = Image.open(uploaded_image)

            # Draw bounding boxes and labels on the image
            annotated_img = draw_boxes_on_image(img)

            # Display the annotated image
            st.image(annotated_img, caption="Processed Image",
                     use_column_width=True)


def draw_boxes_on_image(image):
    img = np.array(image)
    obj = {0: 'L-shaped-couch', 1: 'coffe-table', 2: 'coffee-table', 3: 'drawer',
           4: 'night-stand', 5: 'single-bed', 6: 'tv-cabinet', 7: 'wardrobe'}
    res = model(source=img, imgsz=320, conf=0.15, hide_conf=True, augment=True)
    for i in range(len(res[0].boxes.xyxy)):
        coords = res[0].boxes.xyxy[i]
        print(coords)
        cls = res[0].boxes.cls[i]
        name = obj[int(cls)]
        final = cv2.rectangle(img, (int(coords[0]), int(
            coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
        final = cv2.putText(final, name, (int(coords[0]), int(
            coords[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255, 255), 2)

    return final


if __name__ == "__main__":
    main()
