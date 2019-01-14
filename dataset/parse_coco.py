""" Filter the MSCOCO 2017 dataset and create a simplified version.
    The simplified version is a json array in which each json object contains:
        - Image id
        - Image file path
        - List of categories present in the corresponding image
        - List of captions for the corresponding image
"""


import os
import json
import pickle
import argparse


def group_categories(categories_file):
    """ Group categories by image
    """
    # map each category id to its name
    id_to_category = {}
    for category in categories_file['categories']:
        id_to_category[category['id']] = category['name']

    image_categories = {}
    for category in categories_file['annotations']:
        if category['image_id'] not in image_categories:
            image_categories[category['image_id']] = []
        if id_to_category[category['category_id']] not in image_categories[category['image_id']]:
            image_categories[category['image_id']].append(id_to_category[category['category_id']])
    return image_categories


def group_captions(captions):
    """ Group captions by image """
    image_captions = {}
    for caption in captions:
        img_id = caption['image_id']
        if not img_id in image_captions:
            image_captions[img_id] = []
        image_captions[img_id].append(caption['caption'])
    return image_captions


def get_filename(images):
    """ Get filename of each image """
    image_file = {}
    for image in images:
        image_file[image['id']] = os.path.join('train2017', image['file_name'])
    return image_file


def map_category_id(category_map):
    """ Assign an ID to each category """
    category_id = {}
    id_category = {}
    counter = 0
    for category in category_map:
        category_id[category['name']] = counter
        id_category[counter] = category['name']
        counter += 1
    return category_id, id_category


def save_dataset(image_categories, image_captions, image_file, category_id, id_category, root_dir):
    """ Save parsed dataset """
    print('\nSaving raw dataset...')

    coco_raw = {
        'image_categories': image_categories,
        'image_captions': image_captions,
        'image_file': image_file,
        'category_id': category_id,
        'id_category': id_category
    }

    out_path = '{}/coco_raw.pickle'.format(root_dir)
    pickle_out = open(out_path, 'wb')
    pickle.dump(coco_raw, pickle_out)
    pickle_out.close()

    print('Done.')
    print('\n Data saved to', out_path)


def main(root_dir):
    """ Merge the contents of training and validation dataset into one and
        store only the necessary metadata from each image.
    """
    # load annotations
    print('Loading instances and annotations...')
    captions_file = json.load(open('{}/annotations/captions_train2017.json'.format(root_dir), 'r'))
    categories_file = json.load(open('{}/annotations/instances_train2017.json'.format(root_dir), 'r'))
    print('Done.')

    # group categories by image
    image_categories = group_categories(categories_file)

    # group captions by image
    image_captions = group_captions(captions_file['annotations'])

    # get filename of each image
    image_file = get_filename(captions_file['images'])

    # assign each category an id.
    # we are not using the default ids given in the dataset because
    # the id ranges are not continuous.
    category_id, id_category = map_category_id(categories_file['categories'])
    
    # save parsed coco dataset
    save_dataset(image_categories, image_captions, image_file, category_id, id_category, root_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MSCOCO dataset')
    parser.add_argument(
        '--root', default=os.path.dirname(os.path.realpath(__file__)),
        help='Root directory containing the dataset folders'
    )
    args = parser.parse_args()

    main(args.root)
