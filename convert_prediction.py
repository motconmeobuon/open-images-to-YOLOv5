import os
import shutil
import argparse

import pandas as pd
from tqdm import tqdm

import utils

parser = argparse.ArgumentParser(description='Convert Open Images annotations into YOLOv5 format')
parser.add_argument('-i', '--images', default=None, required=True, help='path image folder', type=str)
parser.add_argument('-a', '--annontations', default=None, required=True, help='path annontations box file', type=str)
parser.add_argument('-c', '--classes', default=None, required=True, help='path class description file', type=str)
parser.add_argument('-f', '--final', default=None, required=True, help='path final folder', type=str)

args = parser.parse_args()

CLASS_ID = {}
CLASS_NUM = {}
# python convert_prediction.py -i ./OID/Dataset/train/ -a ./OID/csv_folder/train-annotations-bbox-example.csv -c ./OID/csv_folder/class-descriptions-boxable-example.csv -f ./YOLOv5-format/train/

if __name__ == '__main__':
    print('[INFO] Get all information...')
    print(args.images)
    path_class = [f.path for f in os.scandir(args.images) if f.is_dir()]
    path_class.sort()
    print(path_class)
    list_class = []

    for i in range(len(path_class)):
        list_class.append(os.path.basename(path_class[i]))
        CLASS_NUM[os.path.basename(path_class[i])] = i
    print(f'List class: {list_class}')

    df_all_class = pd.read_csv(f'{args.classes}', names=['ID', 'Class'])
    index_class = df_all_class.index
    for i in list_class:
        CLASS_ID[i] = df_all_class['ID'][index_class[df_all_class['Class'] == i].tolist()[0]]
    print(CLASS_ID)

    df_all_annontations = pd.read_csv(f'{args.annontations}')
    print('...Done.')

    print('[INFO] Convert Open Images Dataset format to YOLOv5 format...')
    if not os.path.exists(os.path.join(args.final, 'images')):
        os.makedirs(os.path.join(args.final, 'images'))
    final_image_path = os.path.join(args.final, 'images')
    if not os.path.exists(os.path.join(args.final, 'labels')):
        os.makedirs(os.path.join(args.final, 'labels'))
    final_label_path = os.path.join(args.final, 'labels')

    for folder in path_class:
        class_ = folder.split('/')[-1]
        print(class_)
        data_ =  df_all_annontations[df_all_annontations['LabelName'] == CLASS_ID[class_]]
        for file in tqdm(os.listdir(folder)):
            if file.endswith('.jpg'):
                id_image = file.split('.')[0]
                if not os.path.exists(os.path.join(final_image_path, file)):
                    shutil.copyfile(os.path.join(folder, file), os.path.join(final_image_path, file))
                _data = data_[data_['ImageID'] == id_image]
                f = open(os.path.join(final_label_path, f'{id_image}.txt'), 'a')
                for index, row in _data.iterrows():
                    _str = f"{CLASS_NUM[class_]} {round((row['XMin'] + row['XMax']) / 2, 6)} {round((row['YMin'] + row['YMax']) / 2, 6)} {round(row['XMax'] - row['XMin'], 6)} {round(row['YMax'] - row['YMin'], 6)}\n"
                    f.write(_str)
    print('...Done.')

    pass