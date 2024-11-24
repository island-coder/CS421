
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import requests

headers = {
        'User-Agent': 'MMKG_DataCrawler/0.1 (uA5N7w2KEk@protonmail.com)'
    }

def init_grounding_dino():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "IDEA-Research/grounding-dino-base"
    processor_dino = AutoProcessor.from_pretrained(model_id)
    model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return processor_dino,model_dino

def get_groundings(processor_dino,model_dino,image,text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = processor_dino(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model_dino(**inputs)

    results = processor_dino.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image.size[::-1]]
    )

    return results

def draw_bounding_boxes(image, detections):
    # Make a writable copy of the image
    image=np.array(image)  #PIL to np
    image = image.copy()

    # Iterate through each detection
    for detection in detections:
        scores = detection['scores'].cpu().numpy()
        labels = detection['labels']
        boxes = detection['boxes'].cpu().numpy()

        for score, label, box in zip(scores, labels, boxes):
            if score > 0.1:  # You can set a threshold for scores if needed
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, f'{label}: {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 255, 0),
                            thickness=2)


    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def process_groundings(target_entites,article,entity_linker,depiction_handler,processor_dino,model_dino):
    target_text = '. '.join(target_entites)
    for image in article['images']:
        try:
            url=image['url']
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()  
            depiction_image = depiction_handler.add_image(url.split('/')[-1],url)
            image = Image.open(response.raw).convert('RGB')
            plt.imshow(image)
            plt.show()
            grounding_results=get_groundings(processor_dino,model_dino,image,target_text)
            print(grounding_results)
            draw_bounding_boxes(image,grounding_results)
        except Exception as e:
            print(f"Error processing image from {url}: {e}")