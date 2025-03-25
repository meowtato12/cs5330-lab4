# Import libraries
import os
import torch
import numpy as np
import gradio as gr
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Define preset background paths
LAB_PATH = "/content/drive/My Drive/CS5330 - Computer Vision/Lab/Lab 4/backgrounds"
PRESET_BACKGROUNDS = {
    "Campus": f"{LAB_PATH}/campus_sbs.jpg",
    "NEU": f"{LAB_PATH}/sbs_neu.jpg",
    "Steam Clock": f"{LAB_PATH}/side_by_side_steam_clock.jpg",
    "Spatial": f"{LAB_PATH}/spatial_sbs.jpg"
}

"""## Helper Functions: Error Handling and Image Processing"""

# Define depth parameters for stereo composition
DEPTH_PARAMS = {
    'close': {'scale': 1.2, 'disparity': 50},
    'medium': {'scale': 1.0, 'disparity': 25},
    'far': {'scale': 0.7, 'disparity': 10}
}

# Error handling decorator to catch and report errors
def handle_processing_errors(default_return=None):
    """
    Decorator for consistent error handling in image processing functions.
    The default_return should match the expected return signature of the decorated function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = f"Error during {func.__name__}: {str(e)}"

                # Handle different return signatures
                if isinstance(default_return, tuple):
                    return (*default_return, error_msg)
                else:
                    return default_return, error_msg
        return wrapper
    return decorator

# Utility function to align image dimensions
def align_dimensions(img1, img2):
    """Ensures two images have the same dimensions by cropping to the minimum size."""
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    return img1[:h, :w], img2[:h, :w]

"""## Load the model"""

def load_model():
    """Loads the DeepLabV3 model for semantic segmentation."""
    print("Loading Model...")
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device

# Load model
global model, device
model, device = load_model()

"""## Image Segmentation"""

def segment_person(image_input):
    """Segment the person from the input image using semantic segmentation."""
    global model, device

    # Handle different input types
    if isinstance(image_input, str):
        input_image = Image.open(image_input).convert("RGB")
    else:
        input_image = Image.fromarray(image_input).convert("RGB")

    original_image = np.array(input_image)

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    # Obtain the person mask (Category 15 in the COCO dataset represents a person)
    person_mask = (output.argmax(0) == 15).cpu().numpy()

    # Manually create an RGBA image
    height, width = original_image.shape[:2]
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_image[:, :, :3] = original_image  # Copy RGB channels
    rgba_image[:, :, 3] = person_mask * 255  # Set alpha channel

    return rgba_image

"""## Background Handling"""

def load_stereo_image(stereo_path_or_name):
    """Load a stereo image, supporting preset names or file paths."""
    # Check if a preset background is used
    if stereo_path_or_name in PRESET_BACKGROUNDS:
        stereo_path = PRESET_BACKGROUNDS[stereo_path_or_name]
    else:
        stereo_path = stereo_path_or_name

    # Check if the file exists
    if not os.path.exists(stereo_path):
        raise FileNotFoundError(f"Stereo image not found: {stereo_path}")

    # Load the image
    stereo_image = np.array(Image.open(stereo_path).convert("RGB"))
    return stereo_image

def split_side_by_side_image(stereo_image):
    """Split a side-by-side stereo image into left and right images."""
    # Get image dimensions
    height, width = stereo_image.shape[:2]

    # Calculate the midpoint
    mid_point = width // 2

    # Split the image
    left_image = stereo_image[:, :mid_point].copy()
    right_image = stereo_image[:, mid_point:].copy()

    # Ensure both left and right images have the same dimensions
    min_width = min(left_image.shape[1], right_image.shape[1])
    left_image = left_image[:, :min_width]
    right_image = right_image[:, :min_width]

    return left_image, right_image

def insert_person_with_depth(left_image, right_image, segmented_person, depth='medium', x_position=50, y_position=50):
    """Insert the segmented person into left and right stereo images with a specified depth."""
    # Get depth parameters
    depth_params = DEPTH_PARAMS.get(depth, DEPTH_PARAMS['medium'])
    disparity = depth_params['disparity']
    scale_factor = depth_params['scale']

    # Convert the segmented person to PIL and apply scaling
    person_img = Image.fromarray(segmented_person)
    if scale_factor != 1.0:
        new_size = (int(person_img.width * scale_factor),
                   int(person_img.height * scale_factor))
        person_img = person_img.resize(new_size, Image.LANCZOS)

    # Convert background images to PIL
    left_pil = Image.fromarray(left_image)
    right_pil = Image.fromarray(right_image)

    # Calculate person's position (centered on the specified percentages)
    img_width, img_height = left_pil.size
    x_pos = int((x_position / 100) * img_width) - person_img.width // 2
    y_pos = int((y_position / 100) * img_height) - person_img.height // 2

    # Ensure the person stays within image bounds
    x_pos = max(0, min(x_pos, img_width - person_img.width))
    y_pos = max(0, min(y_pos, img_height - person_img.height))

    # Calculate the parallax positions
    left_pos = (x_pos - disparity // 2, y_pos)
    right_pos = (x_pos + disparity // 2, y_pos)

    # Create composite images
    composite_left = left_pil.copy()
    composite_right = right_pil.copy()

    # Paste the person onto both images with alpha blending
    composite_left.paste(person_img, left_pos, person_img)
    composite_right.paste(person_img, right_pos, person_img)

    return np.array(composite_left), np.array(composite_right)

"""## 3D Image Composition"""

def create_anaglyph(left_image, right_image):
    """Create a red-cyan anaglyph image from left and right stereo images."""
    # Ensure images are numpy arrays
    if not isinstance(left_image, np.ndarray):
        left_image = np.array(left_image)
    if not isinstance(right_image, np.ndarray):
        right_image = np.array(right_image)

    # Create an empty stereo image
    anaglyph = np.zeros_like(left_image)

    # Merge channels
    anaglyph[:, :, 0] = left_image[:, :, 0]   # Red channel from the left image
    anaglyph[:, :, 1] = right_image[:, :, 1]  # Green channel from the right image
    anaglyph[:, :, 2] = right_image[:, :, 2]  # Blue channel from the right image

    return anaglyph

def create_side_by_side_preview(left_image, right_image):
    left_aligned, right_aligned = align_dimensions(left_image, right_image)
    return np.hstack((left_aligned, right_aligned))

@handle_processing_errors(default_return=(None, None, None))
def process_preset_background(person_image, background_choice, depth, x_position, y_position):
    """Process 3D image composition using a preset background."""
    if person_image is None:
        return None, None, None, "Please upload a person image"

    # Step 1: Segment the person
    print("Segmenting person...")
    segmented_person = segment_person(person_image)

    # Step 2: Load and split the preset side-by-side stereo image
    print(f"Loading preset background: {background_choice}")
    stereo_image = load_stereo_image(background_choice)
    left_bg, right_bg = split_side_by_side_image(stereo_image)

    # Step 3: Insert the person at different depths
    print(f"Inserting person at depth '{depth}'...")
    composite_left, composite_right = insert_person_with_depth(
        left_bg, right_bg, segmented_person, depth, x_position, y_position
    )

    # Step 4: Create an anaglyph image
    print("Creating red-cyan anaglyph...")
    anaglyph = create_anaglyph(composite_left, composite_right)

    # Create a side-by-side preview of the composite images
    composite_side_by_side = create_side_by_side_preview(composite_left, composite_right)

    # Save the final results
    Image.fromarray(segmented_person).save(os.path.join('output', 'segmented_person.png'))
    Image.fromarray(composite_left).save(os.path.join('output', f'composite_left_{depth}.png'))
    Image.fromarray(composite_right).save(os.path.join('output', f'composite_right_{depth}.png'))
    Image.fromarray(anaglyph).save(os.path.join('output', f'anaglyph_{depth}.png'))
    Image.fromarray(composite_side_by_side).save(os.path.join('output', f'composite_side_by_side_{depth}.png'))

    return segmented_person, composite_side_by_side, anaglyph, "Processing successful!"

"""## User Interface"""

def build_gradio_app():
    with gr.Blocks(title="3D Image Composer") as app:
        gr.Markdown("""
        # 3D Image Composer

        This application allows you to insert a person’s photo into a 3D scene and generate an anaglyph (red-cyan) stereoscopic image.
        """)

        person_input = gr.Image(label="Upload Person Image", type="numpy")

        with gr.Tabs():
            with gr.TabItem("Preset Background"):
                background_choice = gr.Radio(
                    choices=list(PRESET_BACKGROUNDS.keys()),
                    value=list(PRESET_BACKGROUNDS.keys())[0] if PRESET_BACKGROUNDS else None,
                    label="Select Background Scene"
                )

                preset_depth = gr.Radio(
                    choices=["close", "medium", "far"],
                    value="medium",
                    label="Depth Level",
                    info="close = foreground, medium = midground, far = background"
                )

                with gr.Row():
                    x_position = gr.Slider(0, 100, 50, label="Horizontal Position (%)")
                    y_position = gr.Slider(0, 100, 80, label="Vertical Position (%)")

                preset_process_btn = gr.Button("Generate 3D Image with Preset Background", variant="primary")

        with gr.Tabs():
            with gr.TabItem("Segmentation Result"):
                segmented_output = gr.Image(label="Segmented Person Image")

            with gr.TabItem("Stereoscopic Composition"):
                stereo_output = gr.Image(label="Side-by-Side Stereoscopic Image (Composite)")

            with gr.TabItem("3D Effect"):
                gr.Markdown("### View the Following Image Using Red-Cyan 3D Glasses")
                anaglyph_output = gr.Image(label="Anaglyph 3D Image")

        preset_process_btn.click(
            fn=process_preset_background,
            inputs=[person_input, background_choice, preset_depth, x_position, y_position],
            outputs=[segmented_output, stereo_output, anaglyph_output]
        )
    return app

# Main Function
if __name__ == "__main__":
    os.makedirs('backgrounds', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Check if preset backgrounds are available
    missing_backgrounds = []
    for name, path in PRESET_BACKGROUNDS.items():
        if not os.path.exists(path):
            missing_backgrounds.append((name, path))

    if missing_backgrounds:
        print("Warning: The following preset backgrounds were not found:")
        for name, path in missing_backgrounds:
            print(f"  - {name}: {path}")
        print("Please make sure these files are placed in the correct location or update the paths in the PRESET_BACKGROUNDS dictionary.")

    # Launch the application
    app = build_gradio_app()
    app.launch(share=True)