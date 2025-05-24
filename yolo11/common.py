from PIL import Image, ImageDraw, ImageFont

def resize_with_letterbox(image_path, target_size=(640, 640), color=(0, 0, 0)):
    """
    Load an image, resize it to target size while maintaining aspect ratio using letterbox.
    
    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Desired output size (width, height).
        color (tuple): Padding color for letterbox.
    
    Returns:
        Image object: Resized image with letterbox padding.
    """
    img = Image.open(image_path).convert("RGB")
    original_width, original_height = img.size
    target_width, target_height = target_size

    # Calculate scaling factor while maintaining aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize while maintaining aspect ratio
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a blank canvas and paste the resized image (letterbox)
    letterboxed_img = Image.new("RGB", target_size, color)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    letterboxed_img.paste(resized_img, (paste_x, paste_y))

    return letterboxed_img

def draw_boxes(image, boxes, classes, scores):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 15) 
    for box, cl, score in zip(boxes, classes, scores):
        rect = [int(i) for i in box]
        draw.rectangle(rect, outline="red", width=1)
        text = f"{cl} {score:0.2}"
        text_position = rect[:2]
        draw.text(text_position, text, fill="white", font=font)