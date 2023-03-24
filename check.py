import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
def check(image):
    e = torch.load("./models/resnet.pth")
    abd = models.resnet50(pretrained=False)
    abd.load_state_dict(e, strict=False)
    abd.eval()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
    preprocess = transforms.Compose([transforms.Resize(258),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize])
    a = Image.open(image)
    b = preprocess(a).unsqueeze_(0)
    c = abd(b)
    prediction_ResNet50 = torch.argmax(c).item()
    if 151<=prediction_ResNet50 <= 268:
        return True
    return False