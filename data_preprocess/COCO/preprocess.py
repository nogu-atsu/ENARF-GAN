import pickle

import blosc
import cv2
import numpy as np
import skimage.io as io
from pycocotools.coco import COCO
from tqdm import tqdm


def full_body(kpts):
    num_visible_points_per_region = [
        #         [[0, 1, 2, 3, 4], 1],
        [[9, 10], 1],
        [[5, 6, 11, 12], 2],
        [[15, 16], 1]]
    sould_annotated = [5, 6, 11, 12]
    cond = 0
    for region, n_visible in num_visible_points_per_region:
        if np.sum(kpts[region, 2] == 2, axis=0) >= n_visible:
            cond += 1
    if np.all(kpts[sould_annotated, 2] > 0):
        cond += 1
    return cond == 4


def decide_crop_coordinate(kpts):
    """
    kpts: (n_kpts, 3)
    """
    center = (kpts[11, :2] + kpts[12, :2]) / 2
    neck = (kpts[5, :2] + kpts[6, :2]) / 2
    head = neck * 1.5 - center * 0.5
    kpts = kpts[kpts[:, 2] > 0, :2]
    kpts = np.concatenate([head[None], kpts], axis=0)

    #     # center based
    #     kpts = kpts - center
    #     half_size = np.abs(kpts).max()
    #     top_left = center - half_size
    #     bottom_right = center + half_size

    # bbox based
    top_left = kpts.min(axis=0)
    bottom_right = kpts.max(axis=0)
    center = (top_left + bottom_right) / 2
    half_size = np.abs(center - top_left).max() * 1.1

    top_left = center - half_size
    bottom_right = center + half_size
    return top_left.astype("int"), bottom_right.astype("int")


def crop_and_update_annot(img, annot):
    kpts = np.array(an['keypoints']).reshape(-1, 3).copy()
    h, w, _ = img.shape
    top_left, (x2, y2) = decide_crop_coordinate(kpts)
    (x1, y1) = top_left
    cropped = np.pad(img, ((max(0, -y1), max(y2 - h, 0)), (max(0, -x1), max(x2 - w, 0)), (0, 0)), mode="edge")
    cropped = cropped[max(0, y1):max(0, y1) + (y2 - y1), max(0, x1):max(0, x1) + (x2 - x1)]

    kpts[:, :2] -= top_left[None]
    annot['keypoints'] = kpts.reshape(-1).tolist()

    sgm = [np.array(sg).copy().reshape(-1, 2) for sg in annot['segmentation']]
    for sg in sgm:
        sg -= top_left[None]
    annot['segmentation'] = [sg.reshape(-1).tolist() for sg in sgm]
    return cropped, annot


def crop_and_resize(img, top_left, bottom_right, size=128):
    if img.ndim == 2:
        img = img[:, :, None].repeat(3, axis=-1)
    h, w, _ = img.shape
    (x1, y1) = top_left
    (x2, y2) = bottom_right
    cropped = np.pad(img, ((max(0, -y1), max(y2 - h, 0)), (max(0, -x1), max(x2 - w, 0)), (0, 0)), mode="edge")
    cropped = cropped[max(0, y1):max(0, y1) + (y2 - y1), max(0, x1):max(0, x1) + (x2 - x1)]
    cropped = cv2.resize(cropped[:, :, ::-1], (size, size))[:, :, ::-1]
    return cropped


dataDir = '/data/unagi0/noguchi/fiftyone/coco-2017'

count = 0
cropped_img_list = []

dataType = 'train'

for dataType in ["train", "val"]:
    data_dir_name = "validation" if dataType == "val" else dataType
    annFile = '{}/raw/instances_{}2017.json'.format(dataDir, dataType)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # display COCO categories and supercategories
    cats = coco.loadCats(coco.getCatIds())

    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)

    # initialize COCO api for person keypoints annotations
    annFile = '{}/raw/person_keypoints_{}2017.json'.format(dataDir, dataType)
    coco_kps = COCO(annFile)

    for i in tqdm(range(len(imgIds))):
        img = coco.loadImgs(imgIds[i])[0]
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco_kps.loadAnns(annIds)

        I = None
        for an in anns:
            kpts = np.array(an['keypoints']).reshape(-1, 3)
            selected = full_body(kpts)
            if selected:
                top_left, bottom_right = decide_crop_coordinate(kpts)
                size = bottom_right[0] - top_left[0]
                if size >= 128:
                    count += 1
                    if I is None:
                        I = io.imread('%s/%s/data/%s' % (dataDir, data_dir_name, img['file_name']))
                    im = crop_and_resize(I, top_left, bottom_right)
                    cropped_img_list.append(blosc.pack_array(im.transpose(2, 0, 1)))
                    if count % 5000 == 0:
                        print(count)
    print(dataType, "finished")

print(count)
cropped_imgs = np.array(cropped_img_list, dtype="object")

with open("/data/unagi0/noguchi/dataset/COCO/crop128/cache.pickle", "wb") as f:
    pickle.dump(cropped_imgs, f)
