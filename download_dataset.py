import os
import sys
import argparse
import zipfile
import tarfile

from urllib.request import urlretrieve


FLICKR = 'http://press.liacs.nl/mirflickr/mirflickr25k.v2/mirflickr25k.zip'
CIFAR_10 = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


def download_data(dataset_name):

    # Check if the dataset directory exists
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    os.chdir('dataset')

    # Set the url, filename and extract path
    if dataset_name == 'flickr':
        url = FLICKR
        filename = FLICKR.split('/')[-1]
    else:
        url = CIFAR_10
        filename = CIFAR_10.split('/')[-1]
    extract_path, ext = os.path.splitext(filename)

    # Check if the dataset has already been extracted
    if os.path.exists(extract_path):
        print('Dataset %s is already extracted' % extract_path)
        sys.exit(0)

    # Download the dataset
    def download_progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, count * block_size / total_size * 100))
        sys.stdout.flush()

    if os.path.exists(filename):
        print('Dataset %s is already downloaded' % filename)
    else:
        filename, _ = urlretrieve(url, filename, download_progress)
        print('\nSuccessfully Downloaded', filename)

    # Extract the dataset
    if dataset_name == 'flickr':
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall('.')
        zip_ref.close()
    else:
        tar = tarfile.open(filename, "r:gz")
        tar.extractall()
        tar.close()
    print('Successfully extracted.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset to run the CapsNet on it.')
    parser.add_argument('-d', '--dataset', default='flickr', choices=['flickr', 'cifar-10'], help='Dataset Name')
    args = parser.parse_args()

    download_data(args.dataset)
