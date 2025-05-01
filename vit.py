import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['NORMAL', 'PNEUMONIA']
    num_classes = len(class_names)

    # Load processor and model from local directory
    local_model_dir = "./models/vit"
    logger.info(f"Loading ViT processor from {local_model_dir}")
    try:
        processor = ViTImageProcessor.from_pretrained(
            local_model_dir, local_files_only=True
        )
        logger.info("Processor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise

    logger.info(f"Loading ViT model from {local_model_dir}")
    try:
        model = ViTForImageClassification.from_pretrained(
            local_model_dir,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Load fine-tuned checkpoint
    best_checkpoint = "./models/vit.pth"
    logger.info(f"Loading checkpoint from {best_checkpoint}")
    try:
        state_dict = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Could not load checkpoint: {e}")
        logger.warning("Proceeding with pre-trained weights from local directory")

    model = model.to(device)
    return model, processor, class_names

def predict_single_image(image_path, model, processor, class_names, device):
    logger.info(f"Processing image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class = class_names[pred_idx.item()]
    logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence.item():.4f}")
    return predicted_class, confidence.item()


# if __name__ == "__main__":
#     model, processor, class_names = run_inference()
#     # sample_image = "data_new/test/NORMAL/IM-0245-0001.jpeg"
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # model = model.to(device)

#     # prediction, confidence = predict_single_image(
#     #     sample_image, model, processor, class_names, device
#     # )
#     # print(f"Predicted class: {prediction} with confidence: {confidence:.4f}")