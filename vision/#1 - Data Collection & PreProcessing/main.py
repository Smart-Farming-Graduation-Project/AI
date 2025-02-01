from helper_function import *
import warnings
warnings.filterwarnings('ignore')

# Paths Of Files
lbl_dst = ['Data/train/label/', 'Data/test/label/']
img_dst = ['Data/train/images/', 'Data/test/images/']

data_path = ['PlantDoc/TRAIN/', 'PlantDoc/TEST/']
n = 0

CreateHierarchicalFolders(test=True, train=True, val=False)

for path in data_path:
    # Extarct Images & Label Paths
    img_paths, lbl_paths = getPaths(path)
    
    # Extraxt Annotaions
    missing_counter = 0
    all_annotations = []
    for lbl_path in tqdm(lbl_paths):
        filepath = r'{}'.format(lbl_path)
        try :
            all_annotations.append(ExtractLabel(filepath))
        except FileNotFoundError:
            missing_counter += 1
    print('Number Of Missing Image {}'.format(missing_counter))


    all_annotations = [x for xs in tqdm(all_annotations) for x in xs]
    # Convert To DataFrame
    all_annotationsdf = pd.DataFrame(all_annotations, columns = ['filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])

    # Convert Numerical Columns from object to int
    cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
    all_annotationsdf[cols] = all_annotationsdf[cols].astype(float)

    # Remove Ant Object Has Width & Height is zero
    all_annotationsdf.drop(all_annotationsdf[all_annotationsdf['width'] == 0].index.values, inplace=True)

    # Prepare DataFrame To Save
    converted_df = pd.DataFrame(columns=['filename', 'name', 'center_x', 'center_y', 'w', 'h', 'id'])
    for i, row in all_annotationsdf.iterrows():
        width, height, xmin, xmax, ymin, ymax = row['width'], row['height'], row['xmin'], row['xmax'], row['ymin'], row['ymax']
        x_center, y_center, new_width, new_height = Convert_coordinates(width, height, xmin, xmax, ymin, ymax)
        new_row = {'filename': row['filename'], 'name': row['name'], 'center_x': x_center, 'center_y': y_center, 'w': new_width, 'h': new_height, 'id': label_encoding(row['name'])}
        converted_df.loc[len(converted_df)] = new_row

    # Save
    save_image_and_annotation(converted_df, lbl_dst[n], img_dst[n], path)
    n += 1

print('All Done 👌')