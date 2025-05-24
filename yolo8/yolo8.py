import onnxruntime as ort
import numpy as np


def prepare_input(img):
    """
    Function used to convert input image to tensor,
    required as an input to YOLOv8 object detection
    network.
    :param buf: Uploaded file input stream
    :return: Numpy array in a shape (3,width,height) where 3 is number of color channels
    """
    img_width, img_height = img.size
    img = img.convert("RGB")
    input = np.array(img) / 255.0
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, 640, 640)
    return input.astype(np.float32), img_width, img_height


def process_output(output, img_width, img_height):
    """
    Function used to convert RAW output from YOLOv8 to an array
    of detected objects. Each object contain the bounding box of
    this object, the type of object and the probability
    :param output: Raw output of YOLOv8 network which is an array of shape (1,84,8400)
    :param img_width: The width of original image
    :param img_height: The height of original image
    :return: Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
    """
    output = output[0].astype(float)
    output = output.transpose()

    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < 0.5:
            continue
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = ((xc - w/2) / 640) * img_width
        y1 = ((yc - h/2) / 640) * img_height
        x2 = ((xc + w/2) / 640) * img_width
        y2 = ((yc + h/2) / 640) * img_height
        boxes.append([x1, y1, x2, y2, label, prob])

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.5]

    return result


def iou(box1,box2):
    """
    Function calculates "Intersection-over-union" coefficient for specified two boxes
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
    :param box1: First box in format: [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format: [x1,y1,x2,y2,object_class,probability]
    :return: Intersection over union ratio as a float number
    """
    return intersection(box1,box2)/union(box1,box2)


def union(box1,box2):
    """
    Function calculates union area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of the boxes union as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)


def intersection(box1,box2):
    """
    Function calculates intersection area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of intersection of the boxes as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)


# Array of YOLOv8 class labels
yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

#########################################

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


def print_model_inputs(model):
    # Get model inputs
    input_names = [inp.name for inp in model.get_inputs()]
    output_names = [out.name for out in model.get_outputs()]
    print("Model Inputs:", input_names)
    print("Model Outputs:", output_names)

def yolo8m(input):
    model = ort.InferenceSession("yolo8/yolov8m.onnx", providers=['CPUExecutionProvider'])


    outputs = model.run(["output0"], {"images":input})
    return process_output(outputs[0], img_width, img_height)


if __name__ == '__main__':
    image = resize_with_letterbox("street.jpg")

    input, img_width, img_height = prepare_input(image)
    result = yolo8m(input)
    
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20) 
    for row in result:
        rect = [int(i) for i in row[:4]]
        draw.rectangle(rect, outline="red", width=1)
        text = f"{row[4]} {row[5]:0.2}"
        text_position = rect[:2]
        draw.text(text_position, text, fill="white", font=font)

    image.save("output.jpg")  # Save the result
