from dataLoalderForEval import YoloDatasetForEval
from torch.utils.data import DataLoader
from utils.dataloader import YoloDataset, yolo_dataset_collate

if __name__ == '__main__':
    annotation_path = '2007_train.txt'
    with open(annotation_path) as f:
        lines = f.readlines()

    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    train_dataset   = YoloDatasetForEval(lines[:num_train],(416,416))
    gen             = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1, pin_memory=True,collate_fn=yolo_dataset_collate)


    for iteration, batch in enumerate(gen):
        img = batch[0]
        label = batch[1]
        print(img.shape)
        print(label)