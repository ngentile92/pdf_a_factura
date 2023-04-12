from get_data import get_data
import re
import logging
import openai

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data = get_data()



# a function that order desc all the word_count dictionary
def order_desc_word_count(word_count):
    # sort the dictionary by value
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_count


#def remove_repeated_substrings(text):
#    pattern = re.compile(r'(\b\w+\b)(?=.*\1)')
#    return pattern.sub('', text)
## get only the text from the data
#text = [data[key]['text'] for key in data]
#
## process the data text so if it find "DUPLICADO" it removes it
#text = [remove_repeated_substrings(text) for text in text]
#
## 


# Reemplaza esto con tu clave de API personal de OpenAI
openai.api_key = ""

def extract_invoice_information(text):
    CHUNK_SIZE = 2000  # You can adjust the chunk size depending on the average length of your input text
    text_chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    info_dict = {}
    for chunk in text_chunks:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Extrae la siguiente información del siguiente texto, teniendo en cuenta que puede aparecer en diferentes formatos o posiciones dentro del texto: 'Tipo de factura', 'Razón social', 'CUIT', 'Comp. Nro', 'Fecha de Emisión', 'Importe Total' y 'Descripción'. Aquí está el texto de la factura: {chunk}",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )

        extracted_info = response.choices[0].text.strip()

        for line in extracted_info.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)  # Split the line at the first colon
                info_dict[key.strip()] = value.strip()

    return info_dict

#def extract_invoice_information(text):
#    prompt = f"Extrae la siguiente información del siguiente texto, teniendo en cuenta que puede aparecer en diferentes formatos o posiciones dentro del texto:\n\n{text}\n\n"
#    prompt += "1. Tipo de factura: Identifica si es tipo A, B, C, o ninguno. (Por ejemplo: 'Tipo de factura: A')\n"
#    prompt += "2. Razón social: La razón social de quien emitió la factura. (Por ejemplo: 'Razón social: Example Company S.A.')\n"
#    prompt += "3. CUIT: El número de CUIT de quien emitió la factura. (Por ejemplo: 'CUIT: 30-12345678-9')\n"
#    prompt += "4. Número de comprobante: El número de comprobante de quien emitió la factura. (Por ejemplo: 'Comp. Nro: 0001-00012345')\n"
#    prompt += "5. Fecha de emisión: La fecha de emisión de la factura. (Por ejemplo: 'Fecha de Emisión: 01/01/2023')\n"
#    prompt += "6. Importe total: El importe total a pagar. (Por ejemplo: 'Importe Total: $1000')\n"
#    prompt += "7. Descripción: La descripción provista en la factura del concepto. (Por ejemplo: 'Descripción: Servicio de consultoría informática')\n\n"
#
#    response = openai.Completion.create(
#        engine="text-davinci-002",
#        prompt=prompt,
#        max_tokens=150,
#        n=1,
#        stop=None,
#        temperature=0.5,
#    )
#
#    extracted_info = response.choices[0].text.strip()
#
#    # Organiza la información extraída en un diccionario
#    info_dict = {}
#    for line in extracted_info.split("\n"):
#        key, value = line.split(": ")
#        info_dict[key] = value
#
#    return info_dict

def extract_invoice_information_from_files(files_dict):
    result = {}
    for file_name, file_info in files_dict.items():
        invoice_text = file_info['text']
        invoice_info = extract_invoice_information(invoice_text)
        result[file_name] = invoice_info
    return result

def display_invoice_information(extracted_info):
    for file_name, invoice_info in extracted_info.items():
        print(f"\nInformation from {file_name}:")
        for key, value in invoice_info.items():
            print(f"{key}: {value}")

#if __name__ == "__main__":
#    data = get_data()
#    logger.info("Extracting invoice information from files...")
    #extracted_info = extract_invoice_information_from_files(data)
    #logger.info("Displaying extracted information...")
    #display_invoice_information(extracted_info)
    #save extracted info to a .csv file
    #import pandas as pd
    #df = pd.DataFrame(extracted_info)
    #df.to_csv('extracted_info.csv')
    

