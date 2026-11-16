import os
import json
import torch

from PIL import Image
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms


class DomainTagGenerator:
    def __init__(self, model_path, num_classes=3, device=None):
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.num_classes = num_classes

        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.num_classes
        )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            label = predicted.item()

        return label


if __name__ == '__main__':

    # 直接在这里修改路径
    DTG_path = '/home/ubuntu/fuxiwen/opensdi/SAM3Adapter/ckpt/DTG.pth'
    image_folder = '/home/ubuntu/fuxiwen/opensdi/dataset/photoshop/korus/image'
    output_json = './json/korus.json'

    DTG = DomainTagGenerator(
        model_path=DTG_path
    )

    image_extensions = (
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.tif',
        '.tiff'
    )

    label_count = {
        0: 0,
        1: 0,
        2: 0
    }

    cls_dict = {}

    image_files = [
        file_name
        for file_name in os.listdir(image_folder)
        if file_name.lower().endswith(image_extensions)
    ]

    image_files.sort()

    for file_name in image_files:

        image_path = os.path.join(
            image_folder,
            file_name
        )

        label = DTG.predict(image_path)

        cls_dict[file_name] = label
        label_count[label] += 1

    with open(output_json, 'w') as f:
        json.dump(
            cls_dict,
            f,
            indent=4
        )

    print("\nLabel Statistics:")
    print("Label 0:", label_count[0])
    print("Label 1:", label_count[1])
    print("Label 2:", label_count[2])
    print("Total:", sum(label_count.values()))

    print("\nJSON saved to:", output_json)