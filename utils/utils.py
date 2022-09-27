import PIL.Image as pil
import matplotlib.pyplot as plt
from torchvision import transforms
import yaml


def read_image(image_path, size=None, crop=False):
    """

    :param image_path: path to test image
    :param size: resize to which size (H, W)
    :param crop: whether crop the center of the image
    :return: resized image
    """
    input_color = pil.open(image_path).convert('RGB')
    original_width, original_height = input_color.size
    if size is None:
        height, width = original_height, original_width
    else:
        height, width = size[0], size[1]

    left = (original_width - width) / 2
    top = (original_height - height) / 2
    right = (original_width + width) / 2
    bottom = (original_height + height) / 2

    if crop is True:
        input_color = input_color.crop((left, top, right, bottom))
    else:
        input_color = input_color.resize((width, height), pil.LANCZOS)
    input_color = transforms.ToTensor()(input_color).unsqueeze(0)

    return input_color


def show_image(input):
    if len(input.shape) is 4:
        image = input[0]
    else:
        image = input
    img = transforms.ToPILImage()(image)
    img.show()


class Option:
    def __init__(self):
        self.model = {}
        self.loss = {}
        self.training = {}
        self.evaluation = {}

        self.num_dataset = {
            "Training": 0,
            "Evaluation": 0,
        }

    def read(self, path):
        with open(path, 'r') as f:
            configs = yaml.safe_load_all(f.read())

            for config in configs:
                for k, v in config.items():
                    if k == 'Model':
                        self.model = v
                    elif k == 'Loss':
                        self.loss = v
                    elif k == 'Training':
                        self.training = v
                        self.num_dataset['Training'] = len(v)
                    elif k == 'Evaluation':
                        self.evaluation = v
                        self.num_dataset['Evaluation'] = len(v)
                    else:
                        raise NotImplementedError('Invalid config: {}'.format(k))
