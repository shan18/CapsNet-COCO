import random
import argparse
import cv2
import pickle
import nltk
import numpy as np
from pycocotools.coco import COCO

from utils import print_progress_bar


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return word_to_vec_map


def create_embeddings(word_to_vec_map, sentence):
    words = nltk.word_tokenize(sentence.lower())
    avg = np.zeros((50,))
    for word in words:
        try:
            avg += word_to_vec_map[word]
        except KeyError:
            avg += word_to_vec_map['unk']
    avg /= len(words)
    return avg


def map_image_to_caption(coco):
    """
    key: image id
    value: a caption of the image
    """
    image_to_caption = {}
    for img_id in coco.getImgIds():
        annotation_id = coco.getAnnIds(img_id)[random.randint(0, 4)]  # Take any one out of 5 captions
        caption = coco.loadAnns(annotation_id)[0]['caption']
        image_to_caption[img_id] = caption.lower()
    return image_to_caption


def create_dataset(coco, word_to_vec_map, image_to_caption, img_size, dataset_size):
    data = []

    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    image_to_caption_sampled = random.sample(image_to_caption.items(), dataset_size)
    for img_id, caption in image_to_caption_sampled:
        # load image array
        img = coco.loadImgs([img_id])[0]
        img_array = cv2.imread('%s/%s/%s' % (data_dir, data_type, img['file_name']), cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, (img_size, img_size))
        
        # load caption embeddings
        caption_vector = create_embeddings(word_to_vec_map, caption)
        
        data.append((new_img_array, caption_vector))

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
    """ Map image to caption
    :param img_size: Image dimensions
    :param dataset_type: Can be train, val, test
    :param output_dir: Directory where the parsed dataset is to be stored
    """
    coco = COCO(ann_file)  # Initialize coco api
    word_to_vec_map = read_glove_vecs(data_dir + '/glove.6B.50d.txt')  # load embeddings
    print('\nMapping image to caption...')
    image_to_caption = map_image_to_caption(coco)
    print('Done.')

    """ Create dataset """
    print('\nCreating %s dataset...' % dataset_type)
    dataset_size = 10000 if dataset_type == 'train' else 2500
    data = create_dataset(coco, word_to_vec_map, image_to_caption, img_size, dataset_size)
    x, y = create_features_and_labels(data)
    print('Done.')
    save_dataset(x, y, dataset_type, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset using MSCOCO for CapsNet')
    parser.add_argument('-s', '--size', default=250, help='Image size to use in dataset')
    parser.add_argument('-t', '--type', choices=['train', 'val', 'test'], help='Type of dataset')
    parser.add_argument('-o', '--output', default='dataset', help='Directory for storing the preprocessed dataset')
    args = parser.parse_args()

    if args.size < 100 or args.size > 300:
        parser.error('Image size should be within 100 to 300 pixels')

    # specify dataset and annotation directories
    data_dir = 'dataset'
    data_type = str(args.type) + '2017'
    ann_file = '{}/annotations/captions_{}.json'.format(data_dir, data_type)

    main(args.size, args.type, args.output)
