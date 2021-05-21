import os
import cv2
import csv
import logging
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from smoke.modeling.heatmap_coder import (
    get_transfrom_matrix,
    affine_transform,
    gaussian_radius,
    draw_umich_gaussian,
)
from smoke.modeling.smoke_coder import encode_label
from smoke.structures.params_3d import ParamsList
from smoke.data.datasets.data_augmentation import DataAugmentation

TYPE_ID_CONVERSION = {
    'Car': 0,
    'Cyclist': 1,
    'Pedestrian': 2,
}


class KITTIDataset(Dataset):
    def __init__(self, cfg, root, is_train=True, transforms=None):
        super(KITTIDataset, self).__init__()
        self.root = root
        self.image_dir = os.path.join(root, "image_2")
        self.image3_dir = os.path.join(root, "image_3")
        self.label_dir = os.path.join(root, "label_2")
        self.calib_dir = os.path.join(root, "calib")

        self.split = cfg.DATASETS.TRAIN_SPLIT if is_train else cfg.DATASETS.TEST_SPLIT
        self.is_train = is_train
        self.transforms = transforms

        if self.split == "train":
            imageset_txt = os.path.join(root, "ImageSets", "train.txt")
        elif self.split == "val":
            imageset_txt = os.path.join(root, "ImageSets", "val.txt")
        elif self.split == "trainval":
            imageset_txt = os.path.join(root, "ImageSets", "trainval.txt")
        elif self.split == "test":
            imageset_txt = os.path.join(root, "ImageSets", "test.txt")
        else:
            raise ValueError("Invalid split!")

        image_files = []
        for line in open(imageset_txt, "r"):
            base_name = line.replace("\n", "")
            image_name = base_name + ".png"
            image_files.append(image_name)
        self.image_files = image_files
        self.label_files = [i.replace(".png", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)
        self.classes = cfg.DATASETS.DETECT_CLASSES

        self.flip_prob = cfg.INPUT.FLIP_PROB_TRAIN if is_train else 0
        self.aug_prob = cfg.INPUT.SHIFT_SCALE_PROB_TRAIN if is_train else 0
        self.gaussian_prob = cfg.INPUT.GAUSSIAN_NOISE_PROB_TRAIN if is_train else 0
        self.color_prob = cfg.INPUT.COLOR_CHANGE_PROB_TRAIN if is_train else 0
        self.shift_scale = cfg.INPUT.SHIFT_SCALE_TRAIN
        self.right_prob = cfg.INPUT.USE_RIGHT_PROB_TRAIN if is_train else 0
        self.mosaic_prob = cfg.INPUT.MOSAIC_PROB_TRAIN if is_train else 0

        self.num_classes = len(self.classes)

        self.input_width = cfg.INPUT.WIDTH_TRAIN
        self.input_height = cfg.INPUT.HEIGHT_TRAIN
        self.output_width = self.input_width // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.output_height = self.input_height // cfg.MODEL.BACKBONE.DOWN_RATIO
        self.max_objs = cfg.DATASETS.MAX_OBJECTS

        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing KITTI {} set with {} files loaded".format(self.split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # load default parameter here
        original_idx = self.label_files[idx].replace(".txt", "")
        
        anns, P2, P3 = self.load_annotations(idx)
        anns_list = [anns]

        use_left = True
        if (self.is_train) and (random.random() < self.right_prob):
            use_left = False
            K = P3[:3, :3]
            P = P3
            img_path = os.path.join(self.image3_dir, self.image_files[idx])
        else:
            K = P2[:3, :3]
            P = P2
            img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        img = Image.open(img_path)
        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)
        region_list = [[0, self.output_width, 0, self.output_height]]
        P_list = [P]

        """
        resize, horizontal flip, and affine augmentation are performed here.
        since it is complicated to compute heatmap w.r.t transform.
        """
        flipped = False
        if (self.is_train) and (random.random() < self.flip_prob) and (use_left):
            flipped = True
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = size[0] - center[0] - 1
            P[0, 2] = size[0] - P[0, 2] - 1

        affine = False
        if (self.is_train) and (random.random() < self.aug_prob):
            affine = True
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)

            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)

        if (self.is_train) and (0 < self.mosaic_prob) and not affine and not flipped:
            mosaic_idx = random.choice(range(self.num_samples))
            mosaic_anns, mosaic_P2, mosaic_P3 = self.load_annotations(mosaic_idx)
            if use_left:
                mosaic_P = mosaic_P2
                mosaic_img_path = os.path.join(self.image_dir, self.image_files[mosaic_idx])
            else:
                mosaic_P = mosaic_P3
                mosaic_img_path = os.path.join(self.image3_dir, self.image_files[mosaic_idx])

            mosaic_img = Image.open(mosaic_img_path)
            if mosaic_img.size == img.size:
                anns_list.append(mosaic_anns)
                P_list.append(mosaic_P)

                left_ratio = random.uniform(0.3, 0.7)
                region = [0, left_ratio * self.output_width, 0, self.output_height]
                mosaic_region = [left_ratio * self.output_width, self.output_width, 0, self.output_height]
                region_list = [region, mosaic_region]
                mosaic_img = mosaic_img.crop((int(left_ratio*img.size[0]), 0, mosaic_img.size[0], mosaic_img.size[1]))
                img.paste(mosaic_img, (int(left_ratio*img.size[0]), 0, img.size[0], img.size[1]))

        if (self.is_train) and (random.random() < self.gaussian_prob):
            img = DataAugmentation.randomGaussian(img)
        if (self.is_train) and (random.random() < self.color_prob):
            img = DataAugmentation.randomColor(img)

        center_size = [center, size]
        trans_affine = get_transfrom_matrix(
            center_size,
            [self.input_width, self.input_height]
        )
        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (self.input_width, self.input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )
        
        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        shape = (int(image.shape[1] / 4), int(image.shape[0] / 4))
        image = cv2.resize(image, shape, interpolation = cv2.INTER_AREA)

        trans_mat = get_transfrom_matrix(
            center_size,
            [self.output_width, self.output_height]
        )

        if not self.is_train:
            # for inference we parametrize with original size
            P = np.concatenate((P, np.ones((1, 4), dtype=np.float32)), axis=0)
            P[3, :3] = 0
            target = ParamsList(image_size=size,
                                is_train=self.is_train)
            target.add_field("trans_mat", trans_mat)
            target.add_field("P", P)
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target, original_idx

        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
        p_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        flip_mask = np.zeros([self.max_objs], dtype=np.uint8)

        regression_2d = np.zeros([self.max_objs, 4], dtype=np.float32)
        p_2d_offsets = np.zeros([self.max_objs, 2], dtype=np.float32)
        p_2d_whs = np.zeros([self.max_objs, 2], dtype=np.float32)

        for i in range(len(anns_list)):
            anns = anns_list[i]
            region = region_list[i]
            P = P_list[i]
            for i, a in enumerate(anns):
                a = a.copy()
                cls = a["label"]

                locs = np.array(a["locations"])
                rot_y = np.array(a["rot_y"])
                if flipped:
                    locs[0] *= -1
                    rot_y *= -1

                point, box2d, box3d = encode_label(
                    P, rot_y, a["dimensions"], locs
                )
                point = affine_transform(point, trans_mat)
                box2d[:2] = affine_transform(box2d[:2], trans_mat)
                box2d[2:] = affine_transform(box2d[2:], trans_mat)
                box2d[[0, 2]] = box2d[[0, 2]].clip(0, self.output_width - 1)
                box2d[[1, 3]] = box2d[[1, 3]].clip(0, self.output_height - 1)
                h, w = box2d[3] - box2d[1], box2d[2] - box2d[0]
                center_2d = np.array([box2d[0] + box2d[2], box2d[1] + box2d[3]]) * 0.5 
                
                if (region[0] < point[0] < region[1]) and (region[2] < point[1] < region[3]):
                    cv2.circle(image, (int(point[0]), int(point[1])), 3, (255,0,0),-1)
                    
                    point_int = point.astype(np.int32)
                    p_offset = point - point_int
                    radius = gaussian_radius(h, w)
                    radius = max(0, int(radius))
                    heat_map[cls] = draw_umich_gaussian(heat_map[cls], point_int, radius)

                    cls_ids[i] = cls
                    regression[i] = box3d
                    proj_points[i] = point_int
                    p_offsets[i] = p_offset
                    dimensions[i] = np.array(a["dimensions"])
                    locations[i] = locs
                    rotys[i] = rot_y
                    reg_mask[i] = 1 if not affine else 0
                    flip_mask[i] = 1 if not affine and flipped else 0

                    regression_2d[i] = box2d
                    p_2d_offsets[i] = center_2d - point_int
                    p_2d_whs[i] = np.array([w, h])

        
        cv2.imwrite(os.path.join("/root/SMOKE/debug", original_idx + ".jpg"), image)

        P = np.concatenate((P, np.ones((1, 4), dtype=np.float32)), axis=0)
        P[3, :3] = 0
        target = ParamsList(image_size=img.size,
                            is_train=self.is_train)
        target.add_field("hm", heat_map)
        target.add_field("reg", regression)
        target.add_field("cls_ids", cls_ids)
        target.add_field("proj_p", proj_points)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("rotys", rotys)
        target.add_field("trans_mat", trans_mat)
        target.add_field("P", P)
        target.add_field("reg_mask", reg_mask)
        target.add_field("flip_mask", flip_mask)

        target.add_field("reg_2d", regression_2d)
        target.add_field("p_2d_offsets", p_2d_offsets)
        target.add_field("p_2d_whs", p_2d_whs)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, original_idx

    def load_annotations(self, idx):
        annotations = []
        file_name = self.label_files[idx]
        fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']

        if self.is_train:
            with open(os.path.join(self.label_dir, file_name), 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

                for line, row in enumerate(reader):
                    if row["type"] in self.classes:
                        annotations.append({
                            "class": row["type"],
                            "label": TYPE_ID_CONVERSION[row["type"]],
                            "truncation": float(row["truncated"]),
                            "occlusion": float(row["occluded"]),
                            "alpha": float(row["alpha"]),
                            "dimensions": [float(row['dl']), float(row['dh']), float(row['dw'])],
                            "locations": [float(row['lx']), float(row['ly']), float(row['lz'])],
                            "rot_y": float(row["ry"])
                        })
        # get camera intrinsic matrix K
        with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    P2 = row[1:]
                    P2 = [float(i) for i in P2]
                    P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                    continue
                elif row[0] == 'P3:':
                    P3 = row[1:]
                    P3 = [float(i) for i in P3]
                    P3 = np.array(P3, dtype=np.float32).reshape(3, 4)
                    break
        return annotations, P2, P3
