""" Filter the MSCOCO 2017 dataset and create a simplified version.
    The simplified version is a json array in which each json object contains:
        - Image file path
        - Image id
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
    print('Loading annotations...')
    val = json.load(open('{}/annotations/captions_val2017.json'.format(root_dir), 'r'))
    train = json.load(open('{}/annotations/captions_train2017.json'.format(root_dir), 'r'))
    print('Done.')

    print('\nCreating JSON object...')
    # combine all images and annotations together
    images = val['images'] + train['images']
    annotations = val['annotations'] + train['annotations']

    # group annotations by image
    image_to_annotations = {}
    for annotation in annotations:
        img_id = annotation['image_id']
        if not img_id in image_to_annotations:
            image_to_annotations[img_id] = []
        image_to_annotations[img_id].append(annotation)
    
    # create the json blob
    out = []
    for image in images:
        img_id = image['id']
        
        # coco specific here, they store train/val images separately
        path = 'train2017' if 'train2017' in image['coco_url'] else 'val2017'
        
        json_image = {}
        json_image['file_path'] = os.path.join(path, image['file_name'])
        json_image['id'] = img_id
        
        captions = []
        image_annotations = image_to_annotations[img_id]
        for annotation in image_annotations:
            captions.append(annotation['caption'])
        json_image['captions'] = captions
        out.append(json_image)
    
    # save as a json file
    out_path = '{}/coco_raw.json'.format(root_dir)
    json.dump(out, open(out_path, 'w'))
    print('Done.')
    print('\n Data saved to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MSCOCO dataset')
    parser.add_argument('--root', default='dataset', help='Root directory containing the dataset folders')
    args = parser.parse_args()

    main(args.root)
