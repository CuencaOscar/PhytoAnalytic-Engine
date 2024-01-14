from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os
# import matplotlib.pyplot as plt
from Classifier import Classifier


class ModelTest():

    test_dir = './test'
    test = ImageFolder(test_dir, transform=transforms.ToTensor())
    test_images = sorted(os.listdir(test_dir + '/test'))
    print('Las imagenes de test son: ', test_images)
    img, label = test[0]
    # plt.imshow(img.permute(1, 2, 0))
    # plt.show()
    # print('Label:', test_images[0], ', Predicted:',
    #       Classifier.predict_image(img, Classifier.model))
    predictions = Classifier.predict_image(img, Classifier.model)
    for pred_class, pred_prob in predictions:
        print(f'Predicted: {pred_class}, Probability: {pred_prob:.4f}')
