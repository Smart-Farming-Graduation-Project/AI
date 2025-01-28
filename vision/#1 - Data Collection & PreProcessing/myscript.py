import os
import shutil
import sys
import termcolor
description = """ use script with command line arguments either with argument 'all' to add all directories in current directory
or with a list of directories to include ... e.g:
python main.py dir1 dir2 dir3 ...  """

SEP = os.sep
PARENT = os.getcwd()
DATA_DIR = PARENT + SEP +"Data"

def main():
    folders = sys.argv
    if len(folders) == 1:
        raise Exception(termcolor.colored( description  , "red") )
    mkdirIfnotExist(DATA_DIR)
    parents = ["train","test","valid"]
    for parent in parents:
        mkdirIfnotExist( DATA_DIR + SEP + parent)
    subs = ['label' ,'image']
    for sub in subs:
        for parent in parents: 
            mkdirIfnotExist(DATA_DIR + SEP + parent + SEP  + sub)
    folders = folders[1:]
    
    if len(folders) == 1 and folders[0].lower() == 'all':
        folders = [ f.name for f in os.scandir(PARENT) if f.is_dir() and f.name[0] != '.' ]

    folders.remove("Data")

    copy(folders)

    print(termcolor.colored("DONE" , "green"))

def copy(directories:list) -> None:
    """
        Takes a list of directories and searches for train,test,and cross validation directories.
        Copies data inside test,train and valid to 'Current Directory/Data/{test,train,cross validation}'  
    """
    for directory in directories:
        source_dir_path = PARENT + SEP + directory
        if os.path.exists(source_dir_path) and os.path.isdir(source_dir_path):
            copy_train_test_valid(source_dir_path,DATA_DIR)
        else:
            print(termcolor.colored(f"Invalid directory {directory}" , "yellow"))

def copy_train_test_valid(src,dest):
    
    for folder in [f.name for f in os.scandir(src) if f.is_dir()]: 
        if folder == 'train':
            copy_all_inside(src + SEP + "train" , dest+SEP +"train" )
        elif folder == 'test':
            copy_all_inside(src + SEP + "test" , dest+SEP +"test" )
        elif folder == 'valid':
            copy_all_inside(src + SEP + "valid" , dest+SEP +"valid" )
        
def copy_all_inside(src,dst):
    for file in os.listdir(src):
        src_file = src + SEP + file
        if src_file.split('.')[-1] == 'xml':
            dst_file = dst + SEP + 'label'  +SEP+ file
        else:
            dst_file = dst + SEP + 'image'  +SEP+ file

        shutil.copy(src_file , dst_file)
        print(termcolor.colored( f"Copied  {src_file} to {dst_file}" , "green"))

def mkdirIfnotExist(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.mkdir(path)


if __name__ == "__main__":
    main()