from PIL import Image, ImageEnhance
import string
from collections import Counter 
from itertools import tee, count
import pytesseract
from pytesseract import Output
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# https://huggingface.co/docs/transformers/installation
from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
# Torch only works for python 3.7 - 3.9, I learned this the hard way >.<!
import torch
# import asyncio
from time import time
# import icecream as ic

import io, os
from google.cloud import vision

CREDENTIALS =  r'./premium-odyssey-378518-934cec99d0b6.json'

def google_ocr_image_to_text(file_path, CREDENTIALS):
    ''' Function that takes in an image file and returns the ocr text.
    Used in the last step of table extraction, on each of the cropped
    cell images. The cell images should be saved in a directory than input
    into this function sequentially. 
    
    Args:
        file_path: path for cell_images
        CREDENTIALS: sign up for google cloud platform and use your own
    '''
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS
    client = vision.ImageAnnotatorClient()
    
    with io.open(file_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)        
    response = client.document_text_detection(image=image)
    # return response.text_annotations
    if response.text_annotations == []:
        # print('empty cell!')
        return ''
    else:
        return response.text_annotations[0].description

# I have to install Tesseract-OCR in Windows and designate the path
# installation file is found here https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Feng\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def pytess(cell_pil_img):
    return ' '.join(pytesseract.image_to_data(cell_pil_img, output_type=Output.DICT, config='-c tessedit_char_blacklist=œ˜â€œï¬â™Ã©œ¢!|”?«“¥ --psm 6 preserve_interword_spaces')['text']).strip()
    
def uniquify(seq, suffs = count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).
    Credit: https://stackoverflow.com/questions/30650474/python-rename-duplicates-in-list-with-progressive-numbers-without-sorting-list
    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k,v in Counter(seq).items() if v>1] 

    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))  
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix

    return seq

def table_detector(image, THRESHOLD_PROBA):
    '''
    Table detection using DEtect-object TRansformer pre-trained on 1 million tables

    '''

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (model, probas[keep], bboxes_scaled)


def table_struct_recog(image, THRESHOLD_PROBA):
    '''
    Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
    '''

    feature_extractor = DetrFeatureExtractor(do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (model, probas[keep], bboxes_scaled)

def add_padding(pil_img, top, right, bottom, left, color=(255,255,255)):
    '''
    Image padding as part of TSR pre-processing to prevent missing table edges
    '''
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    padded_image = Image.new(pil_img.mode, (new_width, new_height), color)
    padded_image.paste(pil_img, (left, top))
    return padded_image

class TableExtractor():

    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    def plot_results_detection(self, model, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates 
        '''
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax()
            xmin_pad, ymin_pad, xmax_pad, ymax_pad = xmin-delta_xmin, ymin-delta_ymin, xmax+delta_xmax, ymax+delta_ymax 
            # red box shows model bounding box
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,fill=False, color='red', linewidth=1, linestyle='-.'))
            # blue line is after padding
            ax.add_patch(plt.Rectangle((xmin_pad, ymin_pad), xmax_pad - xmin_pad, ymax_pad - ymin_pad, fill=False, color='blue', linewidth=1, linestyle='--'))
            
            text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin-20, ymin-50, text, fontsize=10,bbox=dict(facecolor='yellow', alpha=0.5))
            
        plt.axis('off')
        
    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates 
        '''
        cropped_img_list = []

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin-delta_xmin, ymin-delta_ymin, xmax+delta_xmax, ymax+delta_ymax 
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img_list.append(cropped_img)

        return cropped_img_list

    def generate_structure(self, model, pil_img, prob, boxes):
        """_summary_

        Args:
            model (_type_): ML model for TSR
            pil_img (_type_): cropped table image from TD
            prob (torch.tensor)): tensor with the scores for each label
            boxes (_type_): tensor with bounding box coordinates [xmin, ymin, xmax, ymax]

        Returns:
            _type_: _description_
        """
        
        '''
        Co-ordinates are adjusted here by 3 'pixels'
        To plot table pillow image and the TSR bounding boxes on the table
        '''
        rows = {}
        cols = {}
        idx = 0

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            
            # xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax  # is this line necessary?
            cl = p.argmax() # gets class dictionary 
            class_text = model.config.id2label[cl.item()]

            if class_text == 'table row':
                rows['table row.'+str(idx)] = (xmin, ymin, xmax, ymax)
            if class_text == 'table column':
                cols['table column.'+str(idx)] = (xmin, ymin, xmax, ymax)
            idx += 1
            
        return rows, cols
    
    def plot_TSR(self, model, pil_img, prob, boxes, tsr_fontsize=10):
        idx = 0  # loop index
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        plt.axis('off')
        
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            cl = p.argmax() # gets class dictionary 
            class_text = model.config.id2label[cl.item()]
            
            if class_text == 'table row':  
                if idx % 2 == 0: # even
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=1, linestyle='--'))
                else: # odd
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=1, alpha=0.8))
                text = f'row: {p[cl]:0.2f}'
                ax.text(xmin-10, ymin-10, text, fontsize=tsr_fontsize, bbox=dict(facecolor=self.colors[cl.item()], alpha=0.2), ha='right', rotation=-30)
                
            if class_text == 'table column':
                if idx % 2 == 0: # even
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=1, linestyle='--'))
                else: # odd
                    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=self.colors[cl.item()], linewidth=1, alpha=0.8))
                text = f'column: {p[cl]:0.2f}'
                ax.text(xmin-10, ymin-10, text, fontsize=tsr_fontsize, bbox=dict(facecolor=self.colors[cl.item()], alpha=0.2), ha='left', rotation=30)  
        idx += 1 
        
    def sort_rows_cols(self, rows:dict, cols:dict):
        # Sometimes the header and first row overlap, and we need the header bbox not to have first row's bbox inside the headers bbox
        # sort rows by ymin
        rows_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])}
        # sort cols by xmin
        cols_ = {table_feature : (xmin, ymin, xmax, ymax) for table_feature, (xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}

        return rows_, cols_

    def crop_rows_cols(self, pil_img, rows:dict, cols:dict):
        """
        Extracts the box coordinates of the rows and cols, 
        then crops the rows and cell features into smaller images.

        Args:
        ----------
        pil_img : image file
            cropped table image from table_detector()
        rows, cols : 
            output from generate_structure --> sort_table_features

        Returns:
        -------
        rows, cols :
            added cropped_img as a value in each key
        """
        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax)) # crop the row
            rows[k] = xmin, ymin, xmax, ymax, cropped_img

        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax)) # crop the column
            cols[k] = xmin, ymin, xmax, ymax, cropped_img
            # plt.figure()
            # plt.imshow(cropped_img)
        return rows, cols


    def features_to_cells(self, master_row:dict, cols:dict):
        '''Removes redundant bbox for rows&columns and divides each row into cells from columns
        Args:

        Returns:
        '''
        cells_img = {}
        new_cols = cols
        new_master_row = master_row
        row_idx = 0
        for k_row, v_row in new_master_row.items():
            
            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            # cell_xmin, cell_ymin, cell_xmax, cell_ymax = 0, 0, 0, ymax
            x_offset = v_row[0]  # this variable fixed everything!
            row_img_list = [] # list for collecting cropped row_imgs
            # plt.imshow(row_img)
            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                # xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xmin_col, xmax_col = xmin_col, xmax_col
                cell_xmin = xmin_col - x_offset
                cell_xmax = xmax_col - x_offset
                
                if idx == 0: # at the first column, crop at the left edge of row_img
                    cell_xmin = 0 
                if idx == len(new_cols)-1: # at the last column, crop at the right edge of row_img
                    cell_xmax = xmax
                    
                cell_xmin, cell_ymin, cell_xmax, cell_ymax = cell_xmin, 0, cell_xmax, ymax
                
                # pil_img.crop((xmin, ymin, xmax, ymax))
                row_img_cropped = row_img.crop((cell_xmin, cell_ymin, cell_xmax, cell_ymax)) # the actual cropping
                row_img_list.append(row_img_cropped)

            cells_img[k_row+'.'+str(row_idx)] = row_img_list
            row_idx += 1

        return cells_img, len(new_cols), len(new_master_row)-1

    def clean_dataframe(self, df):
        """ Remove irrelevant symbols that appear with OCR, this is most useful with pytesseract
        """
        for col in df.columns:
            df[col]=df[col].str.replace("|", '', regex=True)
            df[col]=df[col].str.replace("'", '', regex=True)
            df[col]=df[col].str.replace('"', '', regex=True)
            df[col]=df[col].str.replace(']', '', regex=True)
            df[col]=df[col].str.replace('[', '', regex=True)
            df[col]=df[col].str.replace('{', '', regex=True)
            df[col]=df[col].str.replace('}', '', regex=True)
        return df

    def save_df_to_csv(self, df, save_dir):
        return df.to_csv(save_dir, index=False)

    # def create_dataframe(self, c3, cells_pytess_result:list, max_cols:int, max_rows:int):
    def create_dataframe(self, ocr_text_list:list, max_cols:int, max_rows:int, first_row_header=False):
        '''Create dataframe using list of cell values of the table, also checks for valid header of dataframe
        Args:
            cells_pytess_result: list of strings or coroutines, each element representing a cell in a table
            max_cols, max_rows: number of columns and rows
        Returns:
            dataframe : final dataframe after all pre-processing 
        '''
        if first_row_header: # use first row as header
            headers = ocr_text_list[:max_cols] 
            print('HEADERS = {}'.format(headers))
            new_headers = uniquify(headers, (f' {x!s}' for x in string.ascii_lowercase))
            df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)
        else: # use numbers as header
            num_cols = len(ocr_text_list[:max_cols])
            df = pd.DataFrame("", index=range(0, max_rows), columns=list(range(num_cols)))
            # df = pd.DataFrame("", index=range(0, max_rows))
        
        cells_list = ocr_text_list[max_cols:]
        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                # df.iat to access single cell values in a dataframe
                df.iat[nrows, ncols] = str(cells_list[cell_idx]) 
                cell_idx += 1
        
        return df

    # Runs the complete table extraction process of generating pandas dataframes from raw pdf-page images
    def run_extraction(self, image_path:str, cell_img_dir, TD_THRESHOLD=0.6, TSR_THRESHOLD=0.8, 
                       use_pytess=False, print_progress=False, first_row_header=False, autosavecsv = False,
                       show_plots = True, tsr_fontsize=10,
                        delta_xmin=20, delta_ymin=20, delta_xmax=20, delta_ymax=20 ):
        """Runs the complete table extraction process of generating pandas dataframes from raw pdf-page images

        Args:
            image_path (str): file path for input image
            cell_img_dir (_type_): directory path for storing cell images
            TD_THRESHOLD (float, optional): Defaults to 0.6.
            TSR_THRESHOLD (float, optional): Defaults to 0.8.
            use_pytess (bool, optional): set True if you want to use pytess Defaults to False.
            print_progress (bool, optional): Prints process for rows and cols. Defaults to False.
            first_row_header (bool, optional): Dedicate first row to header. Defaults to False.
            tsr_fontsize (int, optional): Fontsize for TSR labels. Defaults to 10.
            autosavecsv: Auto save df to csv. Defaults to False.
            show_plots: Plot results of TD and TSR. Recommend to be False when running multiple images. Defaults to True. 
            delta_xmin (int, optional): Pads xmin for cropping after TD. Defaults to 20.
            delta_ymin (int, optional): Pads ymin for cropping after TD. Defaults to 20.
            delta_xmax (int, optional): Pads xmax for cropping after TD. Defaults to 20.
            delta_ymax (int, optional): Pads ymax for cropping after TD. Defaults to 20.

        Returns:
            None: when no table detected in image
            df_list: list of extracted dataframes 
        """
        image = Image.open(image_path).convert("RGB")
        model, probas, bboxes_scaled = table_detector(image, THRESHOLD_PROBA=TD_THRESHOLD)

        if bboxes_scaled.nelement() == 0: # this ends the run_extraction() function 
            print('No table found in the pdf-page image')
            return None

        if show_plots:
            # plt.ion() 
            self.plot_results_detection(model, image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax) 
        
        cropped_img_list = self.crop_tables(image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)
        
        print('Number of tables in input image: {}'.format(len(cropped_img_list)))
        
        print('Directory for storing cell images:', cell_img_dir)
        
        if (not os.path.exists(cell_img_dir)):
            os.mkdir(cell_img_dir)
        
        # clears directory of all previous cell images
        for f in os.listdir(cell_img_dir):
            os.remove(os.path.join(cell_img_dir, f))
        
        # loop over every image found by table_detector
        df_list = [] # list for collecting extracted dataframes
        for table in cropped_img_list:

            model, probas, bboxes_scaled = table_struct_recog(table, THRESHOLD_PROBA=TSR_THRESHOLD)
            
            rows, cols = self.generate_structure(model, table, probas, bboxes_scaled)
            
            if show_plots:
                self.plot_TSR(model, table, probas, bboxes_scaled, tsr_fontsize=tsr_fontsize)
            
            rows, cols = self.sort_rows_cols(rows, cols)
            
            master_row, cols = self.crop_rows_cols(table, rows, cols)

            cells_img, max_cols, max_rows = self.features_to_cells(master_row, cols)
            
            google_ocr_texts = []  # initialize list for storing google ocr outputs
            google_ocr_coroutines = []  # use this when applying async
            
            print('{} rows'.format(len(rows)))
            print('{} columns'.format(len(cols)))
            print('Total {} cells.'.format(len(rows)*len(cols)))
            
            print('Start Google Vision OCR process...')
            start = time()
            j=0
            for k, img_list in cells_img.items():
                if print_progress:
                    print('-- row: {} -'.format(j))
                
                for i, img in enumerate(img_list):
                    if print_progress:
                        print('col: {}'.format(i))
                        
                    cell_name = '/cell_r{}_c{}.png'.format(j, i)
                    img.save(cell_img_dir + cell_name)
                    
                    # apply google ocr to the newly created cell image
                    google_ocr_texts.append(google_ocr_image_to_text(cell_img_dir + cell_name, CREDENTIALS))
                    # text_google_ocr = google_ocr_image_to_text(cell_img_dir + cell_name, CREDENTIALS)
                    # google_ocr_coroutines.append(text_google_ocr) # store ocr text in list
                j+=1        
                
            # google_ocr_texts = await asyncio.gather(*google_ocr_coroutines)                        
            googleocr_df = self.create_dataframe(google_ocr_texts, max_cols, max_rows, first_row_header=first_row_header) # create dataframe
            # googleocr_df = self.clean_dataframe(googleocr_df)
            print("-- Google OCR Results --")
            print(googleocr_df)
            if autosavecsv:
                self.save_df_to_csv(googleocr_df, 'googleocr_df.csv')  
            print('Google OCR process took {:.2f} seconds'.format(time()-start)) 
                 
            #-------------------------------------------------------------------------            
            if use_pytess:
                sequential_cell_img_list = [] # this is for pytess
                print('Start PyTesseract OCR process...')
                start = time()
                j=0
                for k, img_list in cells_img.items():
                    for i, img in enumerate(img_list):
                        sequential_cell_img_list.append(pytess(img))     
                    j+=1  

                # cells_pytess_result = await asyncio.gather(*sequential_cell_img_list)        
                # pytess_df = self.create_dataframe(cells_pytess_result, max_cols, max_rows)
                
                pytess_df = self.create_dataframe(sequential_cell_img_list, max_cols, max_rows, first_row_header=first_row_header)
                # pytess_df = self.clean_dataframe(pytess_df)
                print("-- Pytesseract Results --")
                print(pytess_df)
                self.save_df_to_csv(pytess_df, 'pytess_df.csv')    
                
                print('PyTesseract OCR process took {:.2f} seconds'.format(time()-start)) 
                print('Errors in OCR is due to either quality of the image or performance of the OCR')
            #-----------------------------------------------------------------------------
            
            df_list.append(googleocr_df)  
        
        if show_plots:
            # plt.ion()     
            plt.show() # I put this in the end to not interrupt the pipeline
        
        print('++-- END TABLE EXTRACTION --++')
        return df_list
        
if __name__ == "__main__":
    
    cell_img_dir = r"C:\Users\Feng\Coding projects\sigtica-table-extraction\cell_images"
    
    data_tables_dir = r"D:\_data_table_images"
    # file_name = '\\0190010c-46.jpg'
    file_name = '\\0806809c-21.jpg' 
    # image_path = data_tables_dir + file_name
    image_path = r'C:\Users\Feng\Coding projects\sagtica-table-extraction\table3x3.jpg'
    # image_path = r'no_table.jpg'
    print('-- image_path --')
    print(image_path)
    
    TD_th = 0.95
    TSR_th = 0.8
    
    TD_padd = 30

    te = TableExtractor()
    if image_path is not None:
        te.run_extraction(image_path, cell_img_dir=cell_img_dir, TD_THRESHOLD=TD_th, TSR_THRESHOLD=TSR_th, 
                                     delta_xmin=TD_padd, delta_ymin=TD_padd, delta_xmax=TD_padd, delta_ymax=TD_padd)