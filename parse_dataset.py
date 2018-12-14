import random
import argparse
import cv2
import pickle
import numpy as np

from pycocotools.coco import COCO


# specify dataset and annotation directories
DATA_DIR = 'dataset'
DATA_TYPE = 'val2017'
ANN_FILE = '{}/annotations/instances_{}.json'.format(DATA_DIR, DATA_TYPE)


def get_categories_and_supercategories(coco):
    """
    Obtain MSCOCO 2017 categories and supercategories
    """
    print('\nObtaining MSCOCO 2017 categories and supercategories...')
    categories = coco.loadCats(coco.getCatIds())
    supercategories = set([category['supercategory'] for category in categories])
    print('Done.')
    return categories, supercategories


def assign_supercategory_ids(supercategories):
    """
    Assign id to each supercategory
    """
    supercategory_ids = {}
    for idx, super_name in enumerate(supercategories):
        supercategory_ids[super_name] = idx
    return supercategory_ids


def map_supercategory_to_image(coco, categories):
    """
    key: supercategory
    value: 'set' of corresponding image ids
    """
    supercategory_to_img = {}
    for category in categories:
        supercategory = category['supercategory']
        category_id = coco.getCatIds(catNms=category['name'])
        img_ids = coco.getImgIds(catIds=category_id)
        if supercategory in supercategory_to_img:
            supercategory_to_img[supercategory] |= set(img_ids)
        else:
            supercategory_to_img[supercategory] = set(img_ids)
    return supercategory_to_img


def map_image_to_supercategories(supercategories, supercategory_to_img, supercategory_ids):
    """
    key: image id
    value: one-hot vector of supercategories present in the image
    """
    image_to_supercategories = {}
    for supercategory, img_ids in supercategory_to_img.items():
        for img_id in img_ids:
            if img_id in image_to_supercategories:
                image_to_supercategories[img_id][supercategory_ids[supercategory]] = 1
            else:
                one_hot = [0] * len(supercategories)
                one_hot[supercategory_ids[supercategory]] = 1
                image_to_supercategories[img_id] = one_hot
    return image_to_supercategories


def create_training_data(coco, image_to_supercategories, img_size):
    training_data = []
    for img_id, supercategory in image_to_supercategories.items():
        img = coco.loadImgs([img_id])[0]
        img_array = cv2.imread('%s/%s/%s' % (DATA_DIR, DATA_TYPE, img['file_name']), cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, (img_size, img_size))
        training_data.append([new_img_array, supercategory])
    random.shuffle(training_data)  # shuffle items to reduce homogeneity
    return training_data


def create_features_and_labels(training_data, img_size):
    X, y = [], []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    # Tweak the feature vector
    X = np.array(X, dtype=np.float32)
    X = X.reshape(-1, img_size, img_size, 1)  # last dimension is to specify grayscale image

    # Tweak the label vector
    y = np.array(y, dtype=np.int64)

    return X, y


def save_dataset(X, y, dataset_type):
    print('\nSaving dataset to disk...')
    X_name = 'X_{dataset_type}.pickle'.format(dataset_type=dataset_type)
    pickle_out_x = open(X_name, 'wb')
    pickle.dump(X, pickle_out_x)
    pickle_out_x.close()

    y_name = 'y_{dataset_type}.pickle'.format(dataset_type=dataset_type)
    pickle_out_y = open(y_name, 'wb')
    pickle.dump(y, pickle_out_y)
    pickle_out_y.close()
    print('Saved.\nFeatures: %s\nLabels: %s' % (X_name, y_name))


def main(img_size, dataset_type):
    """ Map image to supercategories"""
    coco = COCO(ANN_FILE)  # Initialize coco api
    categories, supercategories = get_categories_and_supercategories(coco)
    print('\nMapping image to supercategories...')
    supercategory_ids = assign_supercategory_ids(supercategories)
    supercategory_to_img = map_supercategory_to_image(coco, categories)
    image_to_supercategories = map_image_to_supercategories(supercategories, supercategory_to_img, supercategory_ids)
    print('Done.')

    """ Create dataset """
    print('\nCreating %s dataset...' % dataset_type)
    training_data = create_training_data(coco, image_to_supercategories, img_size)
    X, y = create_features_and_labels(training_data, img_size)
    print('Done.')
    save_dataset(X, y, dataset_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset using MSCOCO for CapsNet')
    parser.add_argument('-s', '--size', default=100, help='Image size to use in dataset')
    parser.add_argument('-t', '--type', choices=['train', 'val', 'test'], help='Type of dataset')
    args = parser.parse_args()

    if args.size < 50 or args.size > 200:
        parser.error('Image size should be within 50 to 200 pixels')

    main(args.size, args.type)
