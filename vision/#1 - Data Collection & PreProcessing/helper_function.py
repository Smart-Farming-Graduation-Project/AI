import glob
import pandas as pd
import shutil
import os
from xml.etree import ElementTree as et
import cv2
import matplotlib.pyplot as plt


# Function To Create Train, Test, Validation Folders
def CreateHierarchicalFolders(ParentFile='Data'):
    '''
        This Function Used To Create Folders To Contain Data In Structure 
        (
            train -> images, label 
            val   -> images, label
            test  -> images, label
        )
    '''
    if not os.path.exists(ParentFile):
        os.makedirs(ParentFile + '/train' + '/images')
        os.makedirs(ParentFile + '/train' + '/label')
        os.makedirs(ParentFile + '/val' + '/images')
        os.makedirs(ParentFile + '/val' + '/label')
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
    img_ext = os.path.splitext(os.listdir(pathfile)[0])[1]
    images_paths = glob.glob(pathfile + '/*' + img_ext)
    label_paths = glob.glob(pathfile + '/*.xml')
    return images_paths, label_paths


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



def save_image_and_annotation(df:pd.DataFrame , dst_annotation_dir , dst_image_dir):
    gr_by = df.groupby("path")
    SEP = os.sep
    for path, groups in gr_by:
        label_path = dst_annotation_dir + SEP + path.split(SEP)[-1][0:-3] + 'txt'
        content = ""
        for i, row in groups.iterrows():
            x_center, y_center, width, height = row['x_center'], row['y_center'], row['new_width'], row['new_height']
            image_id = row["image_id"]
            content += f"{image_id} {width} {height} {x_center} {y_center}"
            if i != len(groups)-1:
                content += "\n"
        save_label(label_path, content)
        shutil.copy(path, dst_image_dir + SEP + path.split(SEP)[-1])
        print("Successfully added")





def save_label(label_path , content):
    with open(label_path,"w+") as file:
        file.write(content)


# This Function Return Detected Image as a array
def Return_Detected_Image(image, boxes):
    """
    Draws multiple bounding boxes and annotations on an image.
    Args:
        image: The input image (numpy array).
        boxes: A list of bounding boxes, where each box is a list of the form:
               [[x_center, y_center, new_width, new_height, annotation], Box2 , .....]
    Returns:
        The image with bounding boxes and annotations drawn.
    """
    img_height, img_width = image.shape[:2]  

    R_color = (255, 0, 0)  
    A_color = (0, 0, 0)    
    thickness = 2          
    font =  cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.7       
    font_thickness = 1     

    for box in boxes:
        x_center, y_center, new_width, new_height, annotation = box

        xmin = int((x_center - new_width / 2) * img_width)
        xmax = int((x_center + new_width / 2) * img_width)
        ymin = int((y_center - new_height / 2) * img_height)
        ymax = int((y_center + new_height / 2) * img_height)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), R_color, thickness)

        (text_width, text_height), _ = cv2.getTextSize(annotation, font, font_scale, font_thickness)
        text_x = xmin
        text_y = ymin - 5 if ymin - 5 > 5 else ymin + 20  

        cv2.rectangle(image, (text_x, text_y - text_height), (text_x + text_width, text_y), R_color, -1)
        
        cv2.putText(image, annotation, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return image



# Display Function used to visulize The Image After Desease Detection with Desease Name 
def Display(image, boxes, display='external'):

    image_with_boxes = Return_Detected_Image(image, boxes)

    if display == 'external':
        window_name = 'Detected Image'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(window_name, 600, 600)  
        cv2.moveWindow(window_name, 150, 50)  
        cv2.imshow(window_name, image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif display == 'internal':
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))  
        plt.axis('off')   
        plt.show()        



 # Main Function       
if __name__ == '__main__':
    pass
