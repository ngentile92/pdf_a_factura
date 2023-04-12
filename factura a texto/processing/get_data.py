import os
import csv
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.clustering import cluster_pdfs
from processing.extract_pdf_text import extract_pdf_text

def process_text(text):
    concatenated_text = ' '.join(text.split('\n'))
    rows = concatenated_text.split('\n')
    # remove extra spaces
    rows = [row.strip() for row in rows]
    # remove empty rows
    data = [[row] for row in rows if row]
    return data

def save_to_csv(results, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, results[0].keys())
        w.writeheader()
        w.writerows(results)

def remove_duplicates(all_data):
    output_list = []
    pattern = r'^(.*?)\bDUPLICADO\b'

    for string in all_data:
        result = re.search(pattern, string[0], re.DOTALL)
        if result:
            output_string = result.group(1)
        else:
            output_string = None
        output_list.append(output_string)

    return output_list

def process_folder(folder_path):
    all_data = []
    i = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            input_pdf = os.path.join(folder_path, filename)
            extracted_text = extract_pdf_text(input_pdf)
            processed_data = process_text(extracted_text)
            for row in processed_data:
                data_row = row[0].split('\t') # Split each row into a list of strings
                all_data.append(data_row)
                # add an id column to the data on the last position
                all_data[-1].append(filename)
                all_data[-1].append(i)
                i += 1
    #all_data = remove_duplicates(all_data)
    return all_data
def get_data(file_path):
    all_data = process_folder(file_path)
    clusters = cluster_pdfs(file_path, 4)
    for data in all_data:
        file_name = data[-2]
        if file_name in clusters:
            data.append(clusters[file_name])
        all_data_dict = {}

    for data in all_data:
        file_name = data[-3]
        cluster_number = data[-1]
        text = data[0]
        all_data_dict[file_name] = {"text": text, "cluster_number": cluster_number}

    return all_data_dict

if __name__ == "__main__":
    file_path = "C:/Users/nagge/Desktop/exa_facturas"
    all_data = get_data(file_path)