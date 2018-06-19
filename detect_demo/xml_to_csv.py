import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):

    xml_list = []

    print("*********",glob.glob(path + '/*.xml'))

    for xml_file in glob.glob(path + '/*.xml'):

        tree = ET.parse(xml_file)

        root = tree.getroot()

        for member in root.findall('object'):

            value = (root.find('filename').text,    #如果生成的xml项后面没有图片格式声明，记得这里加上 .JPG

                     int(root.find('size')[0].text),

                     int(root.find('size')[1].text),

                     member[0].text,

                     int(member[4][0].text),

                     int(member[4][1].text),

                     int(member[4][2].text),

                     int(member[4][3].text)

                     )

            xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    xml_df = pd.DataFrame(xml_list, columns=column_name)

    return xml_df

def main():

    image_path = os.path.join(os.getcwd(), 'annotations')

    image_path = r'E:\detect_demo\train'    #改这里的xml路径

    xml_df = xml_to_csv(image_path)

    xml_df.to_csv('dc_train.csv', index=None)    #生成的csv文件

    print('Successfully converted xml to csv.')

main()

'''
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    # 读取注释文件
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text + '.jpg',
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    # 将所有数据分为样本集和验证集，一般按照3:1的比例
    train_list = xml_list[0: int(len(xml_list) * 0.67)]
    eval_list = xml_list[int(len(xml_list) * 0.67) + 1: ]

    # 保存为CSV格式
    train_df = pd.DataFrame(train_list, columns=column_name)
    eval_df = pd.DataFrame(eval_list, columns=column_name)
    train_df.to_csv('data/train.csv', index=None)
    eval_df.to_csv('data/eval.csv', index=None)


def main():
    path = 'E:/detect_demo/cat'
    xml_to_csv(path)
    print('Successfully converted xml to csv.')

main()
'''
