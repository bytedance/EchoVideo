import os
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import insightface
from insightface.app import FaceAnalysis
from diffusers.utils import load_image
from torchvision.transforms.functional import normalize


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src: tuple, tgt_width: int, tgt_height: int) -> tuple:
    """
    Calculate the resize and crop region for an image to fit into a target width and height while maintaining aspect ratio.

    Args:
        src (tuple): A tuple containing the original height and width of the source image (h, w).
        tgt_width (int): The target width for the resized image.
        tgt_height (int): The target height for the resized image.

    Returns:
        tuple: A tuple containing two tuples:
            - The first tuple represents the top-left corner of the crop region (crop_top, crop_left).
            - The second tuple represents the bottom-right corner of the crop region (crop_bottom, crop_right).
    """
    h, w = src
    r = h / w  # Aspect ratio of the source image

    # Determine the new size while maintaining aspect ratio
    if r > (tgt_height / tgt_width):
        resize_height = tgt_height
        resize_width = int(round(tgt_height / h * w))
    else:
        resize_width = tgt_width
        resize_height = int(round(tgt_width / w * h))

    # Calculate the crop region
    crop_top = int(round((tgt_height - resize_height) / 2.0))
    crop_left = int(round((tgt_width - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def resize_numpy_image_long(image: np.ndarray, resize_long_edge: int = 768):
    """
    Resize the input image to a specified long edge while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): Input image (H x W x C or H x W).
        resize_long_edge (int): The target size for the long edge of the image. Default is 768.

    Returns:
        numpy.ndarray: Resized image with the long edge matching `resize_long_edge`, while maintaining the aspect ratio.
    """
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def img2tensor(imgs, bgr2rgb: bool = True, float32: bool = True):
    """Convert Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change BGR to RGB.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have one element, just return tensor.
    """
    def _totensor(img: np.ndarray, bgr2rgb: bool, float32: bool):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def process_face_embeddings(
    face_helper_1,
    face_helper_2,
    app,
    device,
    weight_dtype,
    image: np.ndarray,
    original_id_image: np.ndarray = None,
    is_align_face: bool = True,
):
    """
    Process face embeddings from an image, extracting relevant features such as face embeddings, landmarks, and parsed
    face features using a series of face detection and alignment tools.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        app: Application instance used for face detection.
        device: Device (CPU or GPU) where the computations will be performed.
        weight_dtype: Data type of the weights for precision (e.g., `torch.float32`).
        image: Input image in RGB format with pixel values in the range [0, 255].
        original_id_image: (Optional) Original image for feature extraction if `is_align_face` is False.
        is_align_face: Boolean flag indicating whether face alignment should be performed.

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding.
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - return_face_features_image_2: Processed face features image after normalization and parsing.
            - face_kps: Keypoints of the face detected in the image.
    """
    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get antelopev2 embedding
    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]
        id_ante_embedding = face_info["embedding"]  # (512,)
        face_kps = face_info["kps"]
    else:
        id_ante_embedding = None
        face_kps = None

    # Using facexlib to detect and align face
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_center_face=True)
    if face_kps is None:
        face_kps = face_helper_1.all_landmarks_5[0]
    if face_kps is not None:
        align_face = align_face_kps(image_bgr, face_kps, face_size=512)
    else:
        print("facexlib align face fail")
        align_face = cv2.resize(image_bgr, (512, 512))  # Sample size for model training

    # In case insightface didn't detect face
    if id_ante_embedding is None:
        print("fail to detect face using insightface, extract embedding on align face")
        id_ante_embedding = face_helper_2.get_feat(align_face)

    id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device, weight_dtype)  # torch.Size([512])
    if id_ante_embedding.ndim == 1:
        id_ante_embedding = id_ante_embedding.unsqueeze(0)  # torch.Size([1, 512])

    # Parsing
    if is_align_face:
        input_tensor = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input_tensor = input_tensor.to(device)
        parsing_out = face_helper_1.face_parse(normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)  # torch.Size([1, 1, 512, 512])
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        black_image = torch.zeros_like(input_tensor)  # torch.Size([1, 3, 512, 512])
        # Only keep the face features
        return_face_features_image_2 = torch.where(bg, black_image, input_tensor)  # torch.Size([1, 3, 512, 512])
    else:
        original_image_bgr = cv2.cvtColor(original_id_image, cv2.COLOR_RGB2BGR)
        input_tensor = img2tensor(original_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0  # torch.Size([1, 3, 512, 512])
        input_tensor = input_tensor.to(device)
        return_face_features_image_2 = input_tensor

    return id_ante_embedding, return_face_features_image_2


def process_face_embeddings_infer(
    face_helper_1,
    face_helper_2,
    app,
    device,
    weight_dtype,
    img_file_path: str,
    is_align_face: bool = True,
):
    """
    Process face embeddings from an input image for inference, including alignment, feature extraction, and embedding
    concatenation.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        app: Application instance used for face detection.
        device: Device (CPU or GPU) where the computations will be performed.
        weight_dtype: Data type of the weights for precision (e.g., `torch.float32`).
        img_file_path: Path to the input image file (string) or a numpy array representing an image.
        is_align_face: Boolean flag indicating whether face alignment should be performed (default: True).

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding.
            - image: Processed face image after feature extraction and alignment.
    """
    # Load and preprocess the input image
    if isinstance(img_file_path, str):
        image = np.array(load_image(image=img_file_path).convert("RGB"))
    else:
        image = np.array(ImageOps.exif_transpose(Image.fromarray(img_file_path)).convert("RGB"))

    # Resize image to ensure the longer side is 1024 pixels
    image = resize_numpy_image_long(image, 1024)
    original_id_image = image

    # Process the image to extract face embeddings and related features
    id_cond, align_crop_face_image = process_face_embeddings(
        face_helper_1,
        face_helper_2,
        app,
        device,
        weight_dtype,
        image,
        original_id_image,
        is_align_face,
    )

    # Convert the aligned cropped face image (torch tensor) to a numpy array
    tensor = align_crop_face_image.cpu().detach()
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)

    return id_cond, tensor


def prepare_face_models(model_path: str, device: str, dtype: torch.dtype):
    """
    Prepare all face models for the facial recognition task.

    Args:
        model_path: Path to the directory containing model files.
        device: The device (e.g., 'cuda', 'cpu') where models will be loaded.
        dtype: Data type (e.g., torch.float32) for model inference.

    Returns:
        Tuple: (face_helper_1, face_helper_2, face_main_model)
    """
    # Get helper model
    face_helper_1 = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        device=device,
        model_rootpath=os.path.join(model_path, "face_encoder"),
    )
    face_helper_1.face_parse = init_parsing_model(
        model_name="bisenet", device=device, model_rootpath=os.path.join(model_path, "face_encoder")
    )
    
    face_helper_2 = insightface.model_zoo.get_model(
        f"{model_path}/face_encoder/models/antelopev2/glintr100.onnx", providers=["CUDAExecutionProvider"]
    )
    face_helper_2.prepare(ctx_id=0)

    # Get local facial extractor part 2
    face_main_model = FaceAnalysis(
        name="antelopev2", root=os.path.join(model_path, "face_encoder"), providers=["CUDAExecutionProvider"]
    )
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))

    # Move face models to device
    face_helper_1.face_det.eval()
    face_helper_1.face_parse.eval()
    face_helper_1.face_det.to(device)
    face_helper_1.face_parse.to(device)

    return face_helper_1, face_helper_2, face_main_model


def resize_image_with_padding(img: Image.Image, target_size=(256, 256)):
    """
    Resize the image while maintaining aspect ratio and add padding.

    Args:
        img (Image.Image): The input image to resize.
        target_size (tuple): The target size for the output image (width, height).

    Returns:
        Image.Image: The resized and padded image.
    """
    # Calculate new size while maintaining aspect ratio
    ratio = min(target_size[0] / img.size[0], target_size[1] / img.size[1])
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.LANCZOS)

    # Create a new image for output, filled with black
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    # Paste the resized image in the center
    new_img.paste(img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    
    return new_img


def align_face_kps(image_np: np.ndarray, landmarks: np.ndarray, face_size: int = 512):
    """
    Align the face based on landmarks.

    Args:
        image_np (np.ndarray): The input image as a NumPy array.
        landmarks (np.ndarray): The facial landmarks for alignment.
        face_size (int): The size of the output aligned face image.

    Returns:
        np.ndarray: The aligned face image as a NumPy array.
    """
    # Define the standard positions of five key points
    standard_landmarks = np.array([
        [30.2946, 51.6963],  # Left eye
        [65.5318, 51.5014],  # Right eye
        [48.0252, 71.7366],  # Nose tip
        [33.5493, 92.3655],  # Left mouth corner
        [62.7299, 92.2041]   # Right mouth corner
    ], dtype=np.float32)
    
    # Scale the key points to face size
    standard_landmarks[:, 0] *= face_size / 96
    standard_landmarks[:, 1] *= face_size / 112

    # Calculate the affine transformation matrix
    matrix = cv2.estimateAffinePartial2D(landmarks, standard_landmarks, method=cv2.LMEDS)[0]
    
    # Apply the affine transformation
    aligned_image = cv2.warpAffine(image_np, matrix, (face_size, face_size), flags=cv2.INTER_LINEAR)
    
    return aligned_image


def pad_image_to_aspect_ratio(image: np.ndarray, ratio: float):
    """
    Pad the image to maintain the specified aspect ratio.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        ratio (float): The target aspect ratio (width / height).

    Returns:
        Image.Image: The padded image as a PIL Image.
    """
    # Get current image dimensions
    current_height, current_width = image.shape[:2]

    # Calculate the shortest side of the current image
    min_side = min(current_height, current_width)

    # Calculate new dimensions based on target aspect ratio
    if ratio > 1.0:
        new_height = min_side
        new_width = int(new_height * ratio)
    else:
        new_width = min_side
        new_height = int(new_width / ratio)

    # Ensure new dimensions are greater than current dimensions
    new_height = max(new_height, current_height)
    new_width = max(new_width, current_width)

    # Calculate padding amounts
    pad_height = new_height - current_height
    pad_width = new_width - current_width

    # Calculate top, bottom, left, and right padding
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Use cv2.copyMakeBorder for padding
    padded_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,  # Padding type
        value=[0, 0, 0]  # Padding color (black)
    )
    
    return Image.fromarray(padded_image)


def extract_face(img_file_path: str, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, aligned_face: bool = False):
    """
    Extract the face from the image.

    Args:
        img_file_path (str): The path to the input image file.
        model_path (str): The path to the directory containing model files.
        device (str): The device (e.g., 'cuda', 'cpu') where models will be loaded.
        dtype (torch.dtype): Data type for model inference.
        aligned_face (bool): Flag indicating whether the face is already aligned.

    Returns:
        Tuple[np.ndarray, torch.Tensor]: The extracted face image as a NumPy array and the face embeddings.
    """
    # Prepare all the face models
    face_helper_1, face_helper_2, face_main_model = prepare_face_models(model_path, device, dtype)
    
    # Prepare model input
    id_cond, image = process_face_embeddings_infer(face_helper_1, face_helper_2,
                                                   face_main_model, device, dtype,
                                                   img_file_path, is_align_face=True)    
    return image, id_cond