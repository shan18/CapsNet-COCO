import os
import random
import h5py
import pickle
import argparse
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # To import function from the utils file
from utils import load_image, print_progress_bar


def load_data(input_path):
    """ Load coco dataset """
    with open(input_path, 'rb') as file:
        coco_raw = pickle.load(file)
    image_categories = coco_raw['image_categories']
    image_file = coco_raw['image_file']
    category_id = coco_raw['category_id']

    return image_categories, image_file, category_id


def encode_images(image_ids, image_file, params):
    """ Store images in a numpy array """

    images = []

    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, params['dataset_size'], prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for image_id in image_ids:
        img_array = load_image(
            os.path.join(params['input_images'], image_file[image_id]),
            size=(params['image_size'], params['image_size']),
            color=params['color']
        )
        images.append(img_array)

        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, params['dataset_size'], prefix = 'Progress:', suffix = 'Complete', length = 50)
        
    return np.array(images)


def encode_categories(image_ids, image_categories, category_id, dataset_size):
    """ Replace all category names with their respective IDs and
        store them in a numpy array as a multi-hot vector.
    """

    categories = []

    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for image_id in image_ids:
        multi_hot = [0] * len(category_id)
        for category in image_categories[image_id]:
            multi_hot[category_id[category]] = 1
        categories.append(multi_hot)

        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    return np.array(categories, dtype=np.int64)


def save_dataset(x, y, out_path):
    """ Save dataset in a '.h5' file """

    path = '{}/capsnet_train_data.h5'.format(out_path)
    h5f = h5py.File(path, 'w')
    h5f.create_dataset('x', data=x)
    h5f.create_dataset('y', data=y)
    h5f.close()

    print('Done.')
    print('Data saved to:', path)


def create_dataset(image_categories, image_file, category_id, params):
    """ Create training dataset """

    image_ids = list(image_categories.keys())
    random.shuffle(image_ids)
    image_ids = image_ids[:params['dataset_size']]

    # encode images
    print('\nEncoding images...')
    x = encode_images(image_ids, image_file, params)
    print('Done.')

    # encode categories
    print('\nEncoding categories...')
    y = encode_categories(image_ids, image_categories, category_id, params['dataset_size'])
    print('Done.')

    # save dataset
    print('\nSaving dataset...')
    save_dataset(x, y, params['output'])


def main(params):
    image_categories, image_file, category_id = load_data(params['input_raw'])

    if len(image_categories) < params['dataset_size']:
        print('Invalid dataset size')
        return

    print('\nCreating and saving dataset...')
    # create and save dataset
    create_dataset(image_categories, image_file, category_id, params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset for training the Capsule Network Model')
    parser.add_argument(
        '--input_raw',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/coco_raw.pickle'),
        help='Path to file containing the raw data'
    )
    parser.add_argument(
        '--input_images',
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset'),
        help='Root directory containing the folders having images'
    )
    parser.add_argument(
        '--output',
        default=os.path.dirname(os.path.realpath(__file__)),
        help='Path to store the dataset'
    )
    parser.add_argument('--dataset_size', default=12500, type=int, help='Size of dataset')
    parser.add_argument('--image_size', default=250, type=int, help='Image size to use in dataset')
    parser.add_argument('--color', action='store_true', help='Images will be stored in BGR format')
    args = parser.parse_args()
    
    params = vars(args)  # convert to dictionary
    main(params)
