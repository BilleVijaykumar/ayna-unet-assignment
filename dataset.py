import os
import json
from PIL import Image
from torch.utils.data import Dataset

class PolygonDataset(Dataset):
    def __init__(self, data_dir, transform=None, color2id=None):
        self.input_dir = os.path.join(data_dir, "inputs")
        self.output_dir = os.path.join(data_dir, "outputs")
        self.mapping = json.load(open(os.path.join(data_dir, "data.json")))
        self.transform = transform
        self.color2id = color2id

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        item = self.mapping[idx]
        input_image = Image.open(os.path.join(self.input_dir, item["input"])).convert("RGB")
        output_image = Image.open(os.path.join(self.output_dir, item["output"])).convert("RGB")
        color = item["color"].lower()
        color_id = self.color2id.get(color, 0)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, color_id, output_image