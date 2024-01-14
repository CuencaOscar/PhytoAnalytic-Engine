from Resnet9 import ResNet9
from DeviceDataLoader import to_device, get_default_device
import torch
import torch.nn.functional as F


class Classifier():

    # Obtener el train
    train = torch.load('./resources/train_dataset.pt')

    device = get_default_device()

    model = to_device(ResNet9(3, len(train.classes)), device)
    model.load_state_dict(torch.load('plant-disease-model.pth'))
    model.eval()

    def predict_image(img, model, topk=3):
        xb = to_device(img.unsqueeze(0), Classifier.device)
        yb = model(xb)
        probs = F.softmax(yb, dim=1)
        top_probs, preds = torch.topk(probs, topk)
        top_probs_percentage = [prob * 100 for prob in top_probs[0]]
        top_classes = [Classifier.train.classes[idx] for idx in preds[0]]

        return list(zip(top_classes, top_probs_percentage))
