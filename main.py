import argparse
import multiprocessing
import os
import xml.etree.ElementTree

from PIL import Image
from pascal_voc_writer import Writer

import config


def yolo2voc(txt_file):
    w, h = Image.open(os.path.join(config.image_dir, f'{txt_file[:-4]}.jpg')).size
    writer = Writer(f'{txt_file[:-4]}.xml', w, h)
    with open(os.path.join(config.label_dir, txt_file)) as f:
        for line in f.readlines():
            label, x_center, y_center, width, height = line.rstrip().split(' ')
            x_min = int(w * max(float(x_center) - float(width) / 2, 0))
            x_max = int(w * min(float(x_center) + float(width) / 2, 1))
            y_min = int(h * max(float(y_center) - float(height) / 2, 0))
            y_max = int(h * min(float(y_center) + float(height) / 2, 1))
            writer.addObject(config.names[int(label)], x_min, y_min, x_max, y_max)
    writer.save(os.path.join(config.label_dir, f'{txt_file[:-4]}.xml'))


def voc2yolo(xml_file):
    in_file = open(f'{config.label_dir}/{xml_file}')

    root = xml.etree.ElementTree.parse(in_file).getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    has_class = False
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name in config.names:
            has_class = True
    if has_class:
        out_file = open(f'{config.label_dir}/{xml_file[:-4]}.txt', 'w')
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name in config.names:
                xml_box = obj.find('bndbox')
                x_min = float(xml_box.find('xmin').text)
                y_min = float(xml_box.find('ymin').text)
                x_max = float(xml_box.find('xmax').text)
                y_max = float(xml_box.find('ymax').text)

                box_x = (x_min + x_max) / 2.0 - 1
                box_y = (y_min + y_max) / 2.0 - 1
                box_w = x_max - x_min
                box_h = y_max - y_min
                box_x = box_x * 1. / w
                box_w = box_w * 1. / w
                box_y = box_y * 1. / h
                box_h = box_h * 1. / h

                b = [box_x, box_y, box_w, box_h]
                cls_id = config.names.index(obj.find('name').text)
                out_file.write(str(cls_id) + " " + " ".join([str(f'{a:.6f}') for a in b]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo2voc', action='store_true', help='YOLO to VOC')
    parser.add_argument('--voc2yolo', action='store_true', help='VOC to YOLO')
    args = parser.parse_args()

    if args.yolo2voc:
        print('YOLO to VOC')
        txt_files = [name for name in os.listdir(config.label_dir) if name.endswith('.txt')]

        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.map(yolo2voc, txt_files)
        pool.close()

    if args.voc2yolo:
        print('VOC to YOLO')
        xml_files = [name for name in os.listdir(config.label_dir) if name.endswith('.xml')]

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pool.map(voc2yolo, xml_files)
        pool.join()
