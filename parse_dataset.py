import random
import argparse
import cv2
import pickle
import numpy as np

from pycocotools.coco import COCO


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


def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """ Call in a loop to create terminal progress bar
    :params iteration: current iteration (Int)
    :params total: total iterations (Int)
    :params prefix: prefix string (Str)
    :params suffix: suffix string (Str)
    :params decimals: positive number of decimals in percent complete (Int)
    :params length: character length of bar (Int)
    :params fill: bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def create_dataset(coco, image_to_supercategories, img_size, dataset_size):
    data = []

    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    image_to_supercategories_sampled = random.sample(image_to_supercategories.items(), dataset_size)  # Take random subset of images
    for img_id, supercategory in image_to_supercategories_sampled:
        img = coco.loadImgs([img_id])[0]
        img_array = cv2.imread('%s/%s/%s' % (data_dir, data_type, img['file_name']), cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, (img_size, img_size))
        data.append([new_img_array, supercategory])

        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    random.shuffle(data)  # shuffle items to reduce homogeneity
    return data


def create_features_and_labels(data):
    """ Create feature and label vector
    :param data: list of (features, labels)
    :return x: Feature vector (numpy array)
    :return y: Label vector (numpy array)
    """
    x, y = [], []
    for features, label in data:
        x.append(features)
        y.append(label)

    # Tweak the feature vector
    x = np.array(x, dtype=np.float32)

    # Tweak the label vector
    y = np.array(y, dtype=np.int64)

    return x, y


def save_dataset(x, y, dataset_type, output_dir):
    """ Save the preprocessed dataset
    :param x: Feature vector
    :param y: Label vector
    :return: a scalar loss value.
    """
    print('\nSaving dataset to disk...')
    x_name = '{out}/x_{dataset_type}.pickle'.format(out=output_dir, dataset_type=dataset_type)
    pickle_out_x = open(x_name, 'wb')
    pickle.dump(x, pickle_out_x)
    pickle_out_x.close()

    y_name = '{out}/y_{dataset_type}.pickle'.format(out=output_dir, dataset_type=dataset_type)
    pickle_out_y = open(y_name, 'wb')
    pickle.dump(y, pickle_out_y)
    pickle_out_y.close()
    print('Saved.\nFeatures: %s\nLabels: %s' % (x_name, y_name))


def main(img_size, dataset_type, output_dir):
    """ Map image to supercategories
    :param img_size: Image dimensions
    :param dataset_type: Can be train, val, test
    :param output_dir: Directory where the parsed dataset is to be stored
    """
    coco = COCO(ann_file)  # Initialize coco api
    categories, supercategories = get_categories_and_supercategories(coco)
    print('\nMapping image to supercategories...')
    supercategory_ids = assign_supercategory_ids(supercategories)
    supercategory_to_img = map_supercategory_to_image(coco, categories)
    image_to_supercategories = map_image_to_supercategories(supercategories, supercategory_to_img, supercategory_ids)
    print('Done.')

    """ Create dataset """
    print('\nCreating %s dataset...' % dataset_type)
    dataset_size = 10000 if dataset_type == 'train' else 2500
    data = create_dataset(coco, image_to_supercategories, img_size, dataset_size)
    x, y = create_features_and_labels(data)
    print('Done.')
    save_dataset(x, y, dataset_type, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset using MSCOCO for CapsNet')
    parser.add_argument('-s', '--size', default=50, help='Image size to use in dataset')
    parser.add_argument('-t', '--type', choices=['train', 'val', 'test'], help='Type of dataset')
    parser.add_argument('-o', '--output', default='dataset', help='Directory for storing the preprocessed dataset')
    args = parser.parse_args()

    if args.size < 50 or args.size > 200:
        parser.error('Image size should be within 50 to 200 pixels')

    # specify dataset and annotation directories
    data_dir = 'dataset'
    data_type = str(args.type) + '2017'
    ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)

    main(args.size, args.type, args.output)
