import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class YoloDatasetForEval(Dataset):
    def __init__(self, train_lines, image_size):
        super(YoloDatasetForEval, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.model_image_size = image_size

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_data(self, annotation_line):
        '''获取数据'''

        line = annotation_line.split()
        image = Image.open(line[0])

        '''直接resize进行识别'''
        image_data = image.convert('RGB')
        img_shape = image.size
        image_data = image_data.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)
        box_data = np.array([np.array(list(map(int, box.split(','))),dtype=np.float32) for box in line[1:]])
        box = np.zeros((len(box_data), 5))

        if len(box_data) > 0:
            box_data[:,0] = box_data[:,0]/img_shape[0]
            box_data[:,1] = box_data[:,1]/img_shape[1]
            box_data[:,2] = box_data[:,2]/img_shape[0]
            box_data[:,3] = box_data[:,3]/img_shape[1]

            box[:len(box_data)] = box_data

        return image_data, box


    def __getitem__(self, index):
        lines = self.train_lines
        n = self.train_batches
        index = index % n

        img, boxes = self.get_data(lines[index])

        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))

        boxes = np.array(boxes,dtype=np.float32)
        return tmp_inp, boxes


# DataLoader中collate_fn使用
def yolo_dataset_collate_eval(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


