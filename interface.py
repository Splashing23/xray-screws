# This is a basic gradio interface that allows
# users to upload an image and trigger a prediction
# Not currently entirely working. 71

import gradio as gr
import cv2
import torch # FIX ME: TRY TO IMPORT ONLY NECESSARY STUFF FOR TORCH
from torchvision.models import resnet18
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

label_mapping = {
    0: 'DePuy Spine Inc',
    1: 'Medtronic Inc',
    2: 'Nuvasive',
    3: 'Orthofix Inc',
    4: 'Rti Surgical Inc',
    5: 'Sea Spine Inc',
    6: 'Stryker Spine'
}

checkpoint_path = 'xray-screws\\resnet18_state_dict_final.pth'

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(label_mapping))
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()

transform = Compose([
    Resize(224, 224),
    Normalize(mean=[0.5742756723783263, 0.5742756723783263, 0.5742756723783263], std=[0.19380555643624103, 0.19380555643624103, 0.19380555643624103]),
    ToTensorV2()
])

def predict(im):
    im = transform(image=im['composite'])['image']
    im = im.unsqueeze(0)
    with torch.no_grad():
        prediction = model(im)
    softmax = torch.softmax(prediction, dim=1)
    classification = softmax.argmax(dim=1).item()
    confidence = softmax[0, classification].item()
    return f"Classification: {label_mapping[classification]}\nConfidence: {(confidence * 100):.2f}%"

demo = gr.Interface(fn=predict, inputs=gr.ImageEditor(
    eraser=False,
    brush=False,
    # value={
    #     'background': None,
    #     'layers': ['xray-screws\\115780309_00_0.jpg'],
    #     'composite': None
    #     },
    # layers=False,
    image_mode='RGB',
    sources=("upload", "clipboard")
    ), outputs=["text"], allow_flagging='never')

# def predict_and_display_results():
#     prediction = predict(im)
#     prediction_output.text = f"Classification: {prediction[0]}\nConfidence: {prediction[1]}"

# with gr.Blocks() as demo:
#     with gr.Row():
#         im = gr.ImageEditor(
#             eraser=False,
#             brush=False,
#             sources=("upload", "clipboard")
#         )
#         im_preview = gr.Image()
        
#     # Add a button to trigger the predict method
#     predict_button = gr.Button(value="Predict")

#     # Text output to display prediction results
#     prediction_output = gr.Textbox(value="Waiting for prediction...")

#     predict_button.click(predict_and_display_results)

#     im.change(predict, outputs=im_preview, inputs=im, show_progress="hidden")

if __name__ == "__main__":
    demo.launch(share=True)