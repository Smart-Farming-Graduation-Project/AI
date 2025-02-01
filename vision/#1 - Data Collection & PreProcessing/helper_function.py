import glob
import pandas as pd
import shutil
import os
from xml.etree import ElementTree as et
from tqdm import tqdm

# Function To Create Train, Test, Validation Folders
def CreateHierarchicalFolders(ParentFile='Data', train = True, test = True, val = True):
    '''
        This Function Used To Create Folders To Contain Data In Structure 
        (
            train -> images, label 
            val   -> images, label
            test  -> images, label
        )
    '''
    if not os.path.exists(ParentFile):
        if train :
            os.makedirs(ParentFile + '/train' + '/images')
            os.makedirs(ParentFile + '/train' + '/label')
        if val :
            os.makedirs(ParentFile + '/val' + '/images')
            os.makedirs(ParentFile + '/val' + '/label')
        if test :
            os.makedirs(ParentFile + '/test' + '/images')
            os.makedirs(ParentFile + '/test' + '/label')

        print("All File Created 👌")
        return True
    else:
        print("The Files Already Existed !")
        return False


# Function To Get Label From XML File
def ExtractLabel(filepath):
    '''
        Function To EXtract Annotation From File (XML)
        Parameters -> XML File Path
        Return -> List Contain All Objects In Image
    '''
    # Get Root Of Hierarchical
    
    root = et.parse(filepath).getroot()

    # Get Name Image, Width, Height
    image_name = root.find('filename').text
    width = root.find('size').find('width').text
    height = root.find('size').find('height').text

    # Get Objects, Bounding Box
    objs = root.findall('object')
    parser = []
    for obj in objs:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        parser.append([image_name, width, height,
                      name, xmin, xmax, ymin, ymax])
    return parser

# Convert Function Maps the Five numbers of Cordinates to four Numbers for bounding box


def Convert_coordinates(width, height, xmin, xmax, ymin, ymax):
    '''
    Functionality of this function Converts bounding box coordinates (xmin, xmax, ymin, ymax)
    to normalized center coordinates (x_center, y_center) and relative width/height.
    Returns:
        x_center, y_center, new_width, new_height (all normalized to [0, 1])
    '''
    new_width = (xmax - xmin) / width
    new_height = (ymax - ymin) / height
    x_center = (xmin / width) + (new_width / 2)
    y_center = (ymin / height) + (new_height / 2)

    return x_center, y_center, new_width, new_height

# Function To Get Paths For All Images And Labels In Folder


def getPaths(pathfile):
    '''
        This Function To Collect Paths For All Images And Annotation (xml) file in folder
        Note -> Path Must Pass As Raw String
        parameters -> Take PathFile That Contain images plus annotation 
        return     -> Return List Of Images Paths And Labeld Paths
    '''
    # this line to get the extension of images like (.jpg, .png, ...)
    images_paths = glob.glob(pathfile + '*.jpg')
    label_paths = glob.glob(pathfile + '*.xml')
    return images_paths, label_paths


class_id = {
    'Bell_pepper leaf spot': 1,
    'Potato leaf early blight': 2,
    'Strawberry leaf': 3,
    'grape leaf': 4,
    'grape leaf black rot': 5,
    'Tomato leaf': 6,
    'Bell_pepper leaf': 7,
    'Potato leaf': 8,
    'Peach leaf': 9,
    'Corn leaf blight': 10,
    'Apple Scab Leaf': 11,
    'Cherry leaf': 12,
    'Tomato leaf bacterial spot': 13,
    'Tomato leaf yellow virus': 14,
    'Corn Gray leaf spot': 15,
    'Apple rust leaf': 16,
    'Raspberry leaf': 17,
    'Blueberry leaf': 18,
    'Squash Powdery mildew leaf': 19,
    'Tomato mold leaf': 20,
    'Tomato Early blight leaf': 21,
    'Tomato leaf late blight': 22,
    'Tomato Septoria leaf spot': 23,
    'Tomato leaf mosaic virus': 24,
    'Potato leaf late blight': 25,
    'Apple leaf': 26,
    'Corn rust leaf': 27,
    'Soyabean leaf': 28,
    'Tomato two spotted spider mites leaf': 29
}

#  Encoding Name Of Object
def label_encoding(x):
    global class_id
    return class_id[x]

'''
# Function To Mapping Class Name To ID and vice-versa

Map_Class_To_Number = {}
Map_Number_To_Class = {}
counter = 1


def map_class_to_number(class_name):

    global counter, Map_Class_To_Number, Map_Number_To_Class

    if class_name not in Map_Class_To_Number:
        Map_Class_To_Number[class_name] = counter
        Map_Number_To_Class[counter] = class_name
        counter += 1
    return Map_Class_To_Number[class_name]


def map_number_to_class(number):

    global Map_Number_To_Class

    if number in Map_Number_To_Class:
        return Map_Number_To_Class[number]
    return "Class Not Found"


print(map_class_to_number("A"))
print(map_class_to_number("B"))
print(map_class_to_number("C"))

print("-" * 30)
# if class name came as a list
class_names = ["D", "E", "F", "G"]
for class_name in class_names:
    print(map_class_to_number(class_name))

print("-" * 30)

print(map_number_to_class(1))
print(map_number_to_class(2))
print(map_number_to_class(3))
print(map_number_to_class(4))
'''


def save_image_and_annotation(df:pd.DataFrame , dst_annotation_dir , dst_image_dir, src):
    counter = 1
    gr_by = df.groupby("filename")
    SEP = os.sep
    for path, groups in tqdm(gr_by):
        label_path = dst_annotation_dir + str(counter) + '.txt'
        content = ""
        for i, row in groups.iterrows():
            image_id, x_center, y_center, width, height = row["id"], row['center_x'], row['center_y'], row['w'], row['h']
            content += f"{image_id} {width} {height} {x_center} {y_center}"
            if i != len(groups)-1:
                content += "\n"
        try:
            save_label(label_path, content)
            shutil.copy(src + path, dst_image_dir + SEP + str(counter) + '.jpg')
            print("Successfully added")
            counter += 1
        except :
            pass


def save_label(label_path , content):
    with open(label_path,"w+") as file:
        file.write(content)


def MergePaths(paths_list):
    all_img, all_label = [], []
    for file_path in paths_list:
        img_path, lbl_path = getPaths(file_path)
        all_img.extend(img_path)
        all_label.extend(lbl_path)
    return all_img, all_label


if __name__ == '__main__':
    pass
