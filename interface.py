# This is a basic gradio interface that allows
# users to upload an image and trigger a prediction
# Not currently entirely working.

import gradio as gr

def predict(im):
    # Placeholder for classification and confidence percentage
    classification = "Placeholder Classification"
    confidence = "Placeholder Confidence"
    return classification, confidence

with gr.Blocks() as demo:
    with gr.Row():
        im = gr.ImageEditor(
            type="numpy",
            crop_size="1:1",
            eraser=False,
            brush=False,
            layers=False,
            sources=("upload", "clipboard")
        )
        im_preview = gr.Image()
        
    # Add a button to trigger the predict method
    predict_button = gr.Button(value="Predict")

    # Text output to display prediction results
    prediction_output = gr.Textbox(value="Waiting for prediction...")

    def predict_and_display_results():
        prediction = predict(im)
        prediction_output.text = f"Classification: {prediction[0]}\nConfidence: {prediction[1]}"

    predict_button.click(predict_and_display_results)

    im.change(predict, outputs=im_preview, inputs=im, show_progress="hidden")

if __name__ == "__main__":
    demo.launch()