import torch
import torchvision.transforms as T

color2id = {
    "red": 0, "green": 1, "blue": 2, "yellow": 3, "orange": 4,
    "purple": 5, "pink": 6, "brown": 7, "black": 8, "white": 9,
    "gray": 10, "cyan": 11, "magenta": 12, "lime": 13, "navy": 14,
    "teal": 15, "maroon": 16, "olive": 17, "silver": 18, "gold": 19
}

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model