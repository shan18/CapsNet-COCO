""" Filter the MSCOCO 2017 dataset and create a simplified version.
    The simplified version is a json array in which each json object contains:
        - Image id
        - Image file path
        - List of categories present in the corresponding image
        - List of captions for the corresponding image
"""


import os
import json
import random
import argparse


def main(root_dir):
    """ Merge the contents of training and validation dataset into one and
        store only the necessary metadata from each image. """
    
    # load annotations
    print('Loading instances and annotations...')
    categories_file = json.load(open('{}/annotations/instances_train2017.json'.format(root_dir), 'r'))
    captions_file = json.load(open('{}/annotations/captions_train2017.json'.format(root_dir), 'r'))
    print('Done.')

    print('\nCreating JSON object...')
    images = categories_file['images']
    categories = categories_file['annotations']
    captions = captions_file['annotations']

    # map each category id to its name
    id_to_category = {}
    for category in categories_file['categories']:
        id_to_category[category['id']] = category['name']

    # group categories by image
    image_categories = {}
    for category in categories:
        if category['image_id'] not in image_categories:
            image_categories[category['image_id']] = []
        if id_to_category[category['category_id']] not in image_categories[category['image_id']]:
            image_categories[category['image_id']].append(id_to_category[category['category_id']])

    # group captions by image
    image_captions = {}
    for caption in captions:
        img_id = caption['image_id']
        if not img_id in image_captions:
            image_captions[img_id] = []
        image_captions[img_id].append(caption['caption'])
    
    # create the json blob
    out = []
    for image in images:
        img_id = image['id']
        if img_id in image_categories and img_id in image_captions:
            json_image = {}
            json_image['id'] = img_id
            json_image['file_path'] = os.path.join('dataset/train2017', image['file_name'])
            json_image['categories'] = image_categories[img_id]
            json_image['captions'] = image_captions[img_id]
            out.append(json_image)
    
    # save as a json file
    out_path = '{}/coco_raw.json'.format(root_dir)
    json.dump(out, open(out_path, 'w'))
    print('Done.')
    print('\n Data saved to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MSCOCO dataset')
    parser.add_argument(
        '--root', default=os.path.dirname(os.path.realpath(__file__)),
        help='Root directory containing the dataset folders'
    )
    args = parser.parse_args()

    main(args.root)
