from pycocotools.coco import COCO


# specify dataset and annotation directories
DATA_DIR = 'dataset'
DATA_TYPE = 'val2017'
ANN_FILE = '{}/annotations/instances_{}.json'.format(DATA_DIR, DATA_TYPE)


def get_categories_and_supercategories(coco):
    """
    Obtain MSCOCO 2017 categories and supercategories
    """
    categories = coco.loadCats(coco.getCatIds())
    supercategories = set([category['supercategory'] for category in categories])
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


def create_dataset():
    """
    :return: 1. Dictionary mapping image to a one-hot vector of its supercategories
             2. List of supercategories where the corresponding index of each supercategory is its id
    """
    coco = COCO(ANN_FILE)  # Initialize coco api
    categories, supercategories = get_categories_and_supercategories(coco)
    supercategory_ids = assign_supercategory_ids(supercategories)
    supercategory_to_img = map_supercategory_to_image(coco, categories)
    image_to_supercategories = map_image_to_supercategories(supercategories, supercategory_to_img, supercategory_ids)
    return image_to_supercategories, supercategories

