import glob
import shutil
import os
from xml.etree import ElementTree as et


# Function To Create Train, Test, Validation Folders
def CreateHierarchicalFolders(ParentFile = 'Data'):
    '''
        This Function Used To Create Folders To Contain Data In Structure 
        (
            train -> images, label 
            val   -> images, label
            test  -> images, label
        )
    '''
    if not os.path.exists(ParentFile) :
        os.makedirs(ParentFile + '/train' + '/images')
        os.makedirs(ParentFile + '/train' + '/label')
        os.makedirs(ParentFile + '/val' + '/images')
        os.makedirs(ParentFile + '/val' + '/label')
        os.makedirs(ParentFile + '/test' + '/images')
        os.makedirs(ParentFile + '/test' + '/label')
        print("All File Created 👌")
        return True
    else :
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
            parser.append([image_name, width, height, name, xmin, xmax, ymin, ymax])
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

    return x_center , y_center , new_width , new_height

if __name__ == '__main__':
    pass
    