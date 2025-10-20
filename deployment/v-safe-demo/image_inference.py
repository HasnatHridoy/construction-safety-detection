

import io
import requests
import onnxruntime as ort
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import cv2
import tempfile



# ---------------------------
# Font helper
# ---------------------------
def get_font(size=20):
    font_name = matplotlib.rcParams['font.sans-serif'][0]
    font_path = matplotlib.font_manager.findfont(font_name)
    return ImageFont.truetype(font_path, size)

# ---------------------------
# Colors and classes
# ---------------------------
COLOR_PALETTE = [
    (220, 20, 60),    # Crimson Red
    (0, 128, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 140, 0),    # Dark Orange
    (255, 215, 0),    # Gold
    (138, 43, 226),   # Blue Violet
    (0, 206, 209),    # Dark Turquoise
    (255, 105, 180),  # Hot Pink
    (70, 130, 180),   # Steel Blue
    (46, 139, 87),    # Sea Green
    (210, 105, 30),   # Chocolate
    (123, 104, 238),  # Medium Slate Blue
    (199, 21, 133),   # Medium Violet Red
]

classes = [
    'None','Boots','C-worker','Cone','Construction-hat','Crane',
    'Excavator','Gloves','Goggles','Ladder','Mask','Truck','Vest'
]

CLASS_COLORS = {cls: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, cls in enumerate(classes)}

# ---------------------------
# Image loading
# ---------------------------
def open_image(path):
    """Load image from local path or URL."""
    if path.startswith('http://') or path.startswith('https://'):
        img = Image.open(io.BytesIO(requests.get(path).content))
    else:
        if os.path.exists(path):
            img = Image.open(path)
        else:
            raise FileNotFoundError(f"The file {path} does not exist.")
    return img

# ---------------------------
# Utilities
# ---------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def box_cxcywh_to_xyxy_numpy(x):
    """Convert [cx, cy, w, h] box format to [x1, y1, x2, y2]."""
    x_c, y_c, w, h = np.split(x, 4, axis=-1)
    b = np.concatenate([
        x_c - 0.5 * np.clip(w, a_min=0.0, a_max=None),
        y_c - 0.5 * np.clip(h, a_min=0.0, a_max=None),
        x_c + 0.5 * np.clip(w, a_min=0.0, a_max=None),
        y_c + 0.5 * np.clip(h, a_min=0.0, a_max=None)
    ], axis=-1)
    return b

# ---------------------------
# RTDETR ONNX Inference
# ---------------------------
class RTDETR_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model_path):
        self.ort_session = ort.InferenceSession(onnx_model_path)
        input_info = self.ort_session.get_inputs()[0]
        self.input_height, self.input_width = input_info.shape[2:]

    def _preprocess_image(self, image):
        """Preprocess the input image for inference."""
        
        image = image.resize((self.input_width, self.input_height))
        image = np.array(image).astype(np.float32) / 255.0
        image = ((image - self.MEANS) / self.STDS).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))   # HWC → CHW
        image = np.expand_dims(image, axis=0)    # Add batch
        return image

    def _post_process(self, outputs, origin_height, origin_width, confidence_threshold, max_number_boxes):
        """Post-process raw outputs into scores, labels, and boxes."""
        pred_boxes, pred_logits = outputs
        prob = sigmoid(pred_logits)

        # Flatten and get top-k
        flat_prob = prob[0].flatten()
        topk_indexes = np.argsort(flat_prob)[-max_number_boxes:][::-1]
        topk_values = np.take_along_axis(flat_prob, topk_indexes, axis=0)
        scores = topk_values
        topk_boxes = topk_indexes // pred_logits.shape[2]
        labels = topk_indexes % pred_logits.shape[2]

        # Gather boxes
        boxes = box_cxcywh_to_xyxy_numpy(pred_boxes[0])
        boxes = np.take_along_axis(
            boxes,
            np.expand_dims(topk_boxes, axis=-1).repeat(4, axis=-1),
            axis=0
        )

        # Rescale boxes
        target_sizes = np.array([[origin_height, origin_width]])
        img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
        scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale_fct[0, :]

        # Filter by confidence
        keep = scores > confidence_threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        return scores, labels, boxes

    def annotate_detections(self, image, boxes, labels, scores=None):
        """Draw bounding boxes and class labels, return PIL.Image."""
        draw = ImageDraw.Draw(image)
        font = get_font()

        for i, box in enumerate(boxes.astype(int)):
            cls_id = labels[i]
            cls_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))

            # Draw bounding box
            draw.rectangle(box.tolist(), outline=color, width=3)

            # Label text
            label = f"{cls_name}"
            if scores is not None:
                label += f" {scores[i]:.2f}"

            # Get text size
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            tx, ty = box[0], max(0, box[1] - th - 4)

            # Background rectangle
            padding = 4
            draw.rectangle([tx, ty, tx + tw + 2*padding, ty + th + 2*padding], fill=color)

            # Put text
            draw.text((tx + 2, ty + 2), label, fill="white", font=font)

        return image

    def run_inference(self, image, confidence_threshold=0.2, max_number_boxes=100):
        """Run inference and return annotated PIL image.
           Accepts PIL.Image directly.
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL.Image")

        origin_width, origin_height = image.size

        # Preprocess
        input_image = self._preprocess_image(image)

        # Run model
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: input_image})

        # Post-process
        scores, labels, boxes = self._post_process(
            outputs, origin_height, origin_width,
            confidence_threshold, max_number_boxes
        )

        # Annotate and return
        return self.annotate_detections(image.copy(), boxes, labels, scores)



    def process_video_to_file(self, video_path, max_duration=5, target_fps=5, max_height=640, confidence_threshold=0.25):
        """
        Process video, run inference with confidence threshold, annotate frames,
        and return a temporary MP4 video file path.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
    
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_duration * original_fps))
        skip_every = max(int(original_fps // target_fps), 1)
    
        # Read first frame to get resized frame size
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Cannot read the first frame of the video")
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        if pil_img.height > max_height:
            aspect_ratio = pil_img.width / pil_img.height
            new_height = max_height
            new_width = int(aspect_ratio * new_height)
            pil_img = pil_img.resize((new_width, new_height))
        frame_size = (pil_img.width, pil_img.height)
    
        # Create temporary file for output video
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, target_fps, frame_size)
    
        # Reset capture to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
    
        while frame_idx < frame_count:
            ret, frame = cap.read()
            if not ret:
                break
    
            if frame_idx % skip_every == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
    
                if pil_img.height > max_height:
                    aspect_ratio = pil_img.width / pil_img.height
                    new_height = max_height
                    new_width = int(aspect_ratio * new_height)
                    pil_img = pil_img.resize((new_width, new_height))
    
                # Run inference with confidence threshold
                annotated_pil = self.run_inference(pil_img, confidence_threshold=confidence_threshold)
    
                # Convert back to BGR for OpenCV
                annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
                out.write(annotated_bgr)
    
            frame_idx += 1
    
        cap.release()
        out.release()
    
        return temp_file.name  # path to temporary annotated video




