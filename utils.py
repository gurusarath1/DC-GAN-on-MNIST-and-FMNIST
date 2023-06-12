import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import make_grid
import torchvision

SAVE_IMAGE_DIR = './saved_images/'
UTIL_PRINTS = False

def get_device():
    dev = 'cpu'

    if torch.cuda.is_available():
        dev = 'cuda'

    print(f'Device = {dev}')

    return dev


# Function Source: Book -  Machinelearning with Pytorch and Sklearn - Sbastian .. Pg:514
def string_tokenizer(text: str):
    text = re.sub('<[^>]*>', '', text)

    emoticons = re.findall(
        '(?::|;|=)(?:-)?(?:\)|\(|D|P)',
        text.lower()
    )

    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')

    tokenized = text.split()

    return tokenized


def get_sentences_from_text(text: str):
    pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)

    return pat.findall(text.lower())


# Obtained from coursera course on GANs
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def save_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), file_name='saved_img.png'):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    print(f'Saving images to file {file_name}')
    plt.savefig(file_name, bbox_inches='tight')


def get_image_tensor(image_path, device='cpu', transform=None, add_batch_dim=False, batch_dim_index=0):
    image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).to(device)
    # [channels, height, width].

    image = image / 255.0

    if add_batch_dim:
        image = torch.unsqueeze(image, batch_dim_index)
        # [batch_size, channels, height, width]

    if transform is not None:
        image = transform(image)

    print(image_path, '- ', image.shape, device)

    return image


# Reference: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
def display_image(tensor_image, batch_dim_exist=False, batch_dim_index=0, save_image=False, file_name='saved_img.png'):
    if batch_dim_exist:
        plt.imshow(tensor_image.squeeze(dim=batch_dim_index).permute(1, 2,
                                                                     0))  # remove batch dim and Make the Channel dim last
    else:
        plt.imshow(tensor_image.permute(1, 2, 0))  # Make the Channel dim last

    if save_image:
        plt.savefig(SAVE_IMAGE_DIR + file_name, bbox_inches='tight')
    else:
        plt.show()


def get_numpy_onehot_array(categorical_numpy_array):
    categorical_numpy_array = categorical_numpy_array.astype(int)

    num_categories = np.max(categorical_numpy_array) + 1
    num_data_points = categorical_numpy_array.shape[0]  # Num samples
    if UTIL_PRINTS: print(f'num_categories = {num_categories}  num_data_points={num_data_points}')

    one_hot_array = np.zeros((num_data_points, num_categories))
    one_hot_array[np.arange(categorical_numpy_array.size), categorical_numpy_array] = 1

    return one_hot_array


def shuffle_two_numpy_array(data_x, data_y):
    num_data_points = data_x.shape[0]  # Num samples
    if UTIL_PRINTS: print(f'num_data_points = {num_data_points}')

    assert num_data_points == data_y.shape[0]

    shuffle_idxs = np.random.permutation(num_data_points)
    return data_x[shuffle_idxs], data_y[shuffle_idxs]


def shuffle_two_torch_array(data_x, data_y):
    num_data_points = data_x.shape[0]  # Num samples
    if UTIL_PRINTS: print(f'num_data_points = {num_data_points}')

    assert num_data_points == data_y.shape[0]

    shuffle_idxs = torch.randperm(num_data_points)
    return data_x[shuffle_idxs], data_y[shuffle_idxs]


def get_torch_model_output_size_at_each_layer(model, input_shape=0, input_tensor=None):
    print('=====================================================')
    print(f'get_torch_model_output_size_at_each_layer type = {type(model)} device = {next(model.parameters()).device}')

    # https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
    # We assume that all the model parameters are in a single device
    if not input_tensor:
        assert input_shape != 0
        input_tensor = torch.ones((input_shape), dtype=torch.float32, device=next(model.parameters()).device)

    print('Printing tensor shape after each layer =====================================================')
    print(f'Input Shape = {input_tensor.shape}')
    print('=====================================================')
    for module in model.modules():

        # These two types are redundant. This loop will iterate inside serial object anyway
        if isinstance(module, (nn.Sequential, type(model))):
            continue

        print(module)
        input_tensor = module(input_tensor)
        print(f'Tensor shape = {input_tensor.shape}')
        print('=====================================================')
