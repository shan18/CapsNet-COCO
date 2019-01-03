import os
import json
import random


def main():
    """ Merge the contents of training and validation dataset into one and
        store only the necessary metadata from each image. """
    
    # load annotations
    print('Loading annotations...')
    val = json.load(open('dataset/annotations/captions_val2017.json', 'r'))
    train = json.load(open('dataset/annotations/captions_train2017.json', 'r'))
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
    json.dump(out, open('dataset/coco_raw.json', 'w'))
    print('Done.')
    print('\n Data saved to dataset/coco_raw.json')


if __name__ == '__main__':
    main()
