import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.regex_functions import *
from processing.get_data import *



# FIRST PROCESING OF THE DATA
def extract_info_cluster1(data_dict):
    results = []

    for file_name, file_info in data_dict.items():
        result = create_empty_result(file_info['cluster_number'], file_name)
        text = file_info['text']
        try:
            process_factura_and_razon_social(result, text)
        except:
            print("No se pudo procesar el tipo de factura o la razón social")
        finally:
            results.append(result)
        try:
            process_cuit_comp_nro_fecha_emision_and_importe_total(result, text)
        except:
            print("No se pudo procesar el CUIT, Comp. Nro, Fecha de Emisión o Importe Total")
        try:
            process_descripcion(result, text)
        except:
            print("No se pudo procesar la descripción")

    return results

def extract_info_cluster0(data_dict):
    results = []
    for file_name, file_info in data_dict.items():
        result = create_empty_result(file_info['cluster_number'], file_name)
        text = file_info['text']
        try:
            pasajes(result ,text)
        except:
            print("No se pudo procesar el tipo de factura o la razón social")
        finally:
            results.append(result)
    return results

def extract_info_cluster2(data_dict):
    results = []
    for file_name, file_info in data_dict.items():
        result = create_empty_result(file_info['cluster_number'], file_name)
        text = file_info['text']
        try:
            pasajes(result ,text)
        except:
            print("No se pudo procesar el tipo de factura o la razón social")
        finally:
            results.append(result)
    return results

def create_empty_result(id, filename):
    return {
        'nombre_factura': filename,
        'cluster_number': id,
        'Tipo de factura': None,
        'Razón social': None,
        'CUIT': None,
        'Comp. Nro': None,
        'Fecha de Emisión': None,
        'Importe Total': None,
        'Descripción': None,
    }

# function that run all
def process_pdf(file_path):

    all_data = get_data(file_path)
    cluster0 = {key: value for key, value in all_data.items() if value['cluster_number'] == 0}
    cluster1 = {key: value for key, value in all_data.items() if value['cluster_number'] == 1}
    cluster2 = {key: value for key, value in all_data.items() if value['cluster_number'] == 2}
    cluster3 = {key: value for key, value in all_data.items() if value['cluster_number'] == 3}
    # sava as a .csv all_data
    cluster0 = extract_info_cluster0(cluster0)
    cluster1 = extract_info_cluster1(cluster1)
    cluster2 = extract_info_cluster2(cluster2)
    cluster3 = extract_info_cluster2(cluster3)
    all_data = cluster0 + cluster1 + cluster2 + cluster3
    return all_data


if __name__ == "__main__":
    input_folder = "C:/Users/nagge/Desktop/exa_facturas"

    file_path = "C:/Users/nagge/Desktop/exa_facturas"
    all_data = process_pdf(file_path)

    # save results as a .csv file all_data, rememer is a list of lists
    
    #output_file = os.path.join(output_folder, "facturas.csv")
    #save_to_csv(results, output_file)
#

