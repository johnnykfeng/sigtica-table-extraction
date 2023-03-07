from TableExtraction import *
import io, os
import numpy as np
import pandas as pd

# grab file names from samples directory
samples_directory = r'.\samples'
file_list = []
filenames = []
for filename in os.listdir(samples_directory):
    filenames.append(filename)
    f = os.path.join(samples_directory, filename)
    file_list.append(f)

print('Number of image files in directory: {}'.format(len(file_list)))


te = TableExtractor()
# print(help(te.run_extraction))

# run table extraction on every file in .\samples
for filename, file_path in zip(filenames, file_list):
    fname = filename.split('.',1)[0]
    print(' ')
    print('Filename: {}'.format(fname))
    print('File path: {}'.format(file_path))
     
    cell_img_dir = r".\cell_images"
    df_list = te.run_extraction(file_path, cell_img_dir = cell_img_dir, 
                                TD_THRESHOLD=0.5, 
                                TSR_THRESHOLD=0.8, 
                                show_plots=True)
    
    if (df_list is not None):
        print('Number of tables in image: {}'.format(len(df_list)))
        for df in df_list:
            save_dir = str(fname)+'.csv'
            df.to_csv(save_dir, index=False, header=False)
        





