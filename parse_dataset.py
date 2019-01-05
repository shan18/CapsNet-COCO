import os
import json
import random
import string
import cv2
import h5py
import pickle
import argparse
import numpy as np

from utils import print_progress_bar


def preprocess_captions(data):
    """ Removes punctuations from captions and tokenizes them """
    for image in data:
        captions = []
        for caption in image['captions']:
            table = str.maketrans(dict.fromkeys(string.punctuation))
            processed_caption = str(caption).lower().translate(table).strip().split()
            processed_caption = ['<sos>'] + processed_caption + ['<eos>']
            captions.append(processed_caption)
        image['captions'] = captions
    return data


def build_vocabulary(data, threshold):
    """ Replace words with frequency less than the threshold with the token 'unk' """

    # count up the number of words
    word_count = {}
    for image in data:
        for caption in image['captions']:
            for word in caption:
                    word_count[word] = word_count.get(word, 0) + 1
    
    # build vocabulary
    vocabulary = ['unk', '<sos>', '<eos>', '<pad>']  # unknown, start, end, pad tokens
    vocabulary += [word for word, frequency in word_count.items() if frequency > threshold]

    # print some stats
    total_words = sum(word_count.values())
    infrequent_words = [word for word, frequency in word_count.items() if frequency <= threshold]
    infrequent_count = sum(word_count[word] for word in infrequent_words)
    print('\nTotal words:', total_words)
    print('number of infrequent words: %d/%d = %.2f%%' % (
        len(infrequent_words), len(word_count), len(infrequent_words) * 100.0 / len(word_count))
    )
    print('number of words in vocabulary would be %d' % len(vocabulary))
    print('number of UNKs: %d/%d = %.2f%%' % (
        infrequent_count, total_words, infrequent_count * 100.0 / total_words)
    )

    # Replace infrequent words with the 'unk' token
    for image in data:
        captions = []
        for caption in image['captions']:
            filtered_caption = [word if word_count.get(word, 0) > threshold else 'unk' for word in caption]
            captions.append(filtered_caption)
        image['captions'] = captions
    
    return data, vocabulary


def get_top_n_captions(data, n):
    """ Consider only top n captions with least 'unk' for each image """

    for image in data:
        if len(image['captions']) > n:
            unk_count = []
            for caption in image['captions']:
                unk_count.append((caption.count('unk'), caption))
            image['captions'] = [x[1] for x in sorted(unk_count, key=lambda x: x[0])[:5]]
    return data


def get_caption_vectors(data, max_length, word_to_index):
    """ Replace words in captions by their corresponding indices in the vocabulary """
    for image in data:
        captions = []
        for caption in image['captions']:
            new_caption = [word_to_index[word] for word in caption]
            if len(new_caption) > max_length:  # Clip long captions
                new_caption = new_caption[:max_length]
            else:  # Pad smaller captions with '<pad>'
                new_caption = new_caption + [word_to_index['<pad>']] * (max_length - len(new_caption))
            captions.append(new_caption)
        image['captions'] = captions
    return data


def encode_captions(data, dataset_size):
    """ Store all captions into a single array """
    captions = []

    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for image in data:
        captions.append(image['captions'])

        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    return np.array(captions, dtype=np.int64)


def encode_images(data, params, dataset_size):
    """ Store images in a numpy array """
    images = []

    # Initial call to print 0% progress
    print_progress_bar_counter = 0
    print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for image in data:
        img_array = cv2.imread('%s/%s' % (params['root'], image['file_path']), cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, (params['size'], params['size']))
        images.append(new_img_array)

        # Update Progress Bar
        print_progress_bar_counter += 1
        print_progress_bar(print_progress_bar_counter, dataset_size, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
    return np.expand_dims(np.array(images, dtype=np.float32), axis=-1).astype('float32') / 255.


def create_dataset(data, params):
    """ Split dataset into training validation and test dataset """
    data_train = data[:params['train']]
    data_val = data[params['train']:params['train'] + params['val']]

    # Create training dataset
    print('Encoding training images...')
    x_train = encode_images(data_train, params, params['train'])
    print('Encoding training captions...')
    y_train = encode_captions(data_train, params['train'])
    print('Saving training dataset...')
    save_dataset(x_train, y_train, 'train', params['root'])
    # free up memory so that the program does not freeze
    del x_train, y_train

    # Create validation dataset
    print('\nEncoding validation images...')
    x_val = encode_images(data_val, params, params['val'])
    print('Encoding validation captions...')
    y_val = encode_captions(data_val, params['val'])
    print('Saving validation dataset...')
    save_dataset(x_val, y_val, 'val', params['root'])
    # free up memory so that the program does not freeze
    del x_val, y_val


def save_dataset(x, y, dataset_type, root_path):
    """ Save dataset in a '.h5' file """

    path = '{}/{}_data.h5'.format(root_path, dataset_type)
    h5f = h5py.File(path, 'w')
    h5f.create_dataset('x_{}'.format(dataset_type), data=x)
    h5f.create_dataset('y_{}'.format(dataset_type), data=y)
    h5f.close()

    print('Done.')
    print('{} data saved to:'.format(dataset_type), path)


def save_vocabulary(vocabulary, word_to_index, index_to_word, root_path):
    """ Save vocabulary in a '.pickle' file """
    print('\nSaving vocabulary...')

    vocabulary_dict = {
        'vocabulary': vocabulary,
        'word_to_index': word_to_index,
        'index_to_word': index_to_word
    }

    path = '{}/vocabulary.pickle'.format(root_path)
    pickle_out = open(path, 'wb')
    pickle.dump(vocabulary_dict, pickle_out)
    pickle_out.close()

    print('Done.')
    print('Vocabulary saved to:', path)


def main(params):
    data = json.load(open(params['input'], 'r'))
    random.shuffle(data)

    # Remove punctuations from captions and tokenize them
    print('Removing punctuations from captions...')
    data = preprocess_captions(data)
    print('Done.')

    # Build vocabulary
    print('\nBuilding vocabulary...')
    data, vocabulary = build_vocabulary(data, params['threshold'])
    index_to_word = {i+1: w for i, w in enumerate(vocabulary)}  # a 1-indexed vocab translation table
    word_to_index = {w: i+1 for i, w in enumerate(vocabulary)}  # inverse table
    print('Done.')

    print('\nCreating training and validation dataset...')
    # every image should have consistent number of captions
    data = get_top_n_captions(data, 5)

    # replace words in captions by their corresponding indices in the vocabulary
    data = get_caption_vectors(data, params['length'], word_to_index)

    # create and save training and validation dataset
    create_dataset(data, params)
    print('Done.')
    
    # save vocabulary
    save_vocabulary(vocabulary, word_to_index, index_to_word, params['root'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MSCOCO dataset')

    # Input json
    parser.add_argument('--input', required=True, help='Path to input json file containing the raw data')
    parser.add_argument('--root', default='dataset', help='Root directory containing the dataset folders and input json')

    # Dataset size
    parser.add_argument('--train', default=10000, type=int, help='Number of images to assign to training data')
    parser.add_argument('--val', default=2500, type=int, help='Number of images to assign to validation data')
    parser.add_argument('--test', default=0, type=int, help='Number of images to assign to testing data')

    # Image and caption options
    parser.add_argument('--size', default=250, help='Image size to use in dataset')
    parser.add_argument(
        '--length', default=16, type=int, help='Max number of words in a caption. Captions longer than this get clipped.'
    )
    parser.add_argument(
        '--threshold', default=5, type=int,
        help='Words with frequency more than the threshold will be included in the vocabulary.'
    )

    args = parser.parse_args()

    if args.size < 100 or args.size > 300:  # validation on image size
        parser.error('Image size should be within 100 to 300 pixels')
    
    params = vars(args)  # convert to dictionary
    params['data_dir'] = 'dataset'
    
    main(params)
