import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import inception_v3

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    class_names = ['NORMAL', 'PNEUMONIA']
    num_classes = len(class_names)
    
    print("Loading pre-trained Inception V3 model from PyTorch...")
    model = inception_v3()
    
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    best_checkpoint = 'models/inception.pth'
    try:
        state_dict = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Proceeding with pre-trained weights.")
    
    processor = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.4822, 0.4822, 0.4822], [0.2362, 0.2362, 0.2362])
    ])
    
    return model, processor, class_names

def predict_single_image(image_path, model, processor, class_names, device):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image).unsqueeze(0)  
    
    pixel_values = inputs.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values)
        
        if isinstance(outputs, tuple):
            logits = outputs[0]  
        else:
            logits = outputs
        
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        
    predicted_class = class_names[pred_idx.item()]
    return predicted_class, confidence.item()