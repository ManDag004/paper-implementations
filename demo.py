import gradio as gr
import torch
from models import LeNet5
from PIL import Image, ImageOps
from torchvision import transforms

def load_lenet5():
    model = LeNet5()
    model.load_state_dict(torch.load('Computer Vision/LeNet5/lenet5.pth'))
    model.eval()
    return model

def lenet_app():
    model = load_lenet5()

    def predict_lenet(image):
        image = image['composite']
        preprocess = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        image = preprocess(image).view(1, -1, 28, 28)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        return str(predicted.item())

    return gr.Interface(
        fn=predict_lenet,
        inputs=gr.Sketchpad(
            canvas_size=(280, 280),  # Set the canvas size to 28x28 pixels
            image_mode='L',         # Set image mode to 'L' for grayscale
            label="Draw a digit",
            type="pil",
            brush=gr.Brush(colors=["#ffffff"], color_mode='fixed', default_size=18)
        ),
        outputs="text",
        title="LeNet-5: Handwritten Digit Recognition"
    )

# Main function to run the Gradio interface
def main():
    iface = gr.TabbedInterface(
        [lenet_app(), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text"), gr.Interface(lambda x: x, "text", "text")],
        ["LeNet-5", "Other Model 1", "Other Model 2", "Other Model 1", "Other Model 2", "Other Model 1", "Other Model 2", "Other Model 1", "Other Model 2", "Other Model 1", "Other Model 2", "Other Model 1", "Other Model 2", "Other Model 1", "Other Model 2", "Other Model 1", "Other Model 2", "Other Model 1", "Other Model 2"]
    )

    iface.launch()

if __name__ == "__main__":
    main()