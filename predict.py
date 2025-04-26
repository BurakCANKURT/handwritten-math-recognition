import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from model import MathCNN
import os


class Predict:
    def __init__(self):
        
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Create a Debug Folder 
        os.makedirs("debug_chars", exist_ok=True)

    def preprocess_image(self, image, x, y):
        image = ImageOps.invert(image) 
        image = image.point(lambda x: 0 if x < 110 else 255, '1')  
        bbox = image.getbbox()
        if bbox:
            image = image.crop(bbox)
        image = ImageOps.pad(image, (28, 28), centering=(0.5, 0.5))  

        # Save Segments
        image.save(f"debug_chars/segment_{x}_{y}.png")

        return self.transform(image).unsqueeze(0)

    def segment_and_predict(self, canvas_image, model, device, show_boxes=False):
        img_array = np.array(canvas_image.convert("L"))
        _, thresh = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(thresh, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        

        groups = []
        current_group = []
        THRESHOLD = 35

        for i, cnt in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(cnt)
            current_group.append(cnt)

            if i + 1 < len(sorted_contours):
                next_x, _, next_w, _ = cv2.boundingRect(sorted_contours[i + 1])
                distance = next_x - (x + w)
                

                if distance > THRESHOLD:
                    groups.append(current_group)
                    current_group = []

        if current_group:
            groups.append(current_group)

        full_number_predictions = []
        box_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        for group in groups:
            group_preds = []
            group = sorted(group, key=lambda ctr: cv2.boundingRect(ctr)[0])  

            for cnt in group:
                x, y, w, h = cv2.boundingRect(cnt)
                padding = 10
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w = w + padding * 2
                h = h + padding * 2

                char_img = img_array[y:y+h, x:x+w]
                char_pil = Image.fromarray(char_img)
                char_tensor = self.preprocess_image(char_pil, x, y).to(device)

                with torch.no_grad():
                    output = model(char_tensor)
                    pred = output.argmax(dim=1).item()
                    group_preds.append(str(pred))

                if show_boxes:
                    cv2.rectangle(box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            number_str = "".join(group_preds)
            full_number_predictions.append(number_str)

        if show_boxes:
            box_image = Image.fromarray(cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB))
            return full_number_predictions, box_image
        else:
            return full_number_predictions
