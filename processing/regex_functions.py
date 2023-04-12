import re


def tipo_factura(string):
    pattern = r'^.*?\b([a-zA-Z])\s+Cod\.'
    result = re.search(pattern, string, re.IGNORECASE)
    if result:
        return result.group(1).upper()
    else:
        return None
def razon_social(text):
    pattern = r'Razón Social:\s*(.*?)\s*Domicilio Comercial:'
    match = re.search(pattern, text, re.DOTALL)
    if match and "Fecha" in match.group(): # check if match is not None and "Fecha" is in the matched string
        return extract_before_fecha_emision(match.group()) # use match.group() to extract the matched string
    elif match:
        return match.group(1)
    else:
        return None
    
def extract_before_fecha_emision(string):
    pattern = r'^(.*?)\s*Fecha de Emisión:'
    result = re.search(pattern, string, re.DOTALL)
    if result:
        # eliminata "Razón Social:"
        return result.group(1)[14:].strip()
    else:
        return None
    

def extract_cuit(text):
    pattern = r'(.*?)\s+CUIT'
    match = re.search(pattern, text)
    # get the full text of match.group(1)
    match = extract_last_11_numbers(match.group(1))
    if match:
        return match
    else:
        return None
def extract_last_11_numbers(text):
    pattern = r'\d{11}$'
    # delete - and spaces
    text = text.replace('-', '').replace(' ', '')
    result = re.search(pattern, text)
    if result:
        return result.group(0)
    else:
        return None
def extract_comp_nro(text):
    pattern = r'Comp\. Nro:\s*(\d+)'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        pattern = r'FACTURA\s+(\d+)\s+-\s+(.*?)\s+Fecha'
        match = re.search(pattern, text)
        if match:
            return match.group(2)
        else:
            return None

def extract_fecha_emision(text):
    pattern = r'Fecha de Emisión:\s*(\d{2}/\d{2}/\d{4})'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None

def extract_importe_total(text):
    pattern = r'Importe Total:\s+\$\s*([\d,]+(?:\.\d{1,2})?)'
    match = re.search(pattern, text)
    if match:
        number = match.group(1).replace(",", "").replace(".", "")
        return float(number)/100
    else:
        return None

def extract_between_subtotal_and_unidades(text):
    pattern = r'Subtotal(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = match.group(1).strip()
        if result[0].isdigit():
            result = re.sub(r'^\s*\d*\s*', '', result)
        return result
    else:
        return None
def extract_text_before_number(text):
    pattern = r'(.*?)\d+,'
    result = re.search(pattern, text)
    if result:
        return result.group(1).strip()
    else:
        return None

# delete everythig before a letter
def delete_before_first_letter(text):
    pattern = r'[a-zA-Z].*'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    else:
        return None

def process_factura_and_razon_social(result, string):
    result['Tipo de factura'] = tipo_factura(string)
    result['Razón social'] = razon_social(string)


def process_cuit_comp_nro_fecha_emision_and_importe_total(result, string):
    result['CUIT'] = extract_cuit(string)
    result['Comp. Nro'] = extract_comp_nro(string)
    result['Fecha de Emisión'] = extract_fecha_emision(string)
    result['Importe Total'] = extract_importe_total(string)


def process_descripcion(result, string):
    descripcion = extract_between_subtotal_and_unidades(string)
    descripcion = extract_text_before_number(descripcion)
    descripcion = delete_before_first_letter(descripcion)

    if "Otros" in descripcion[0]:
        regex_pattern = r"\s+Otros.*"
        text_without_extra = re.sub(regex_pattern, "", descripcion)
        descripcion = text_without_extra

    result['Descripción'] = descripcion

def pasaje_info(string):
    pattern = r'\b(\d{1,2}/\d{1,2}/\d{2,4})'
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return None
    
def pasajes(result, string):
    result['Fecha de Emisión'] = pasaje_info(string)