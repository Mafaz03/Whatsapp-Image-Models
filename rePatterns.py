import re

def extract_numbers(single_string):
    pattern = re.compile(r'\((\d+)\)$')
    match = pattern.search(single_string)
    if match: return int(match.group(1))
    else: return None

def extract_and_remove_number(single_string):
    pattern = re.compile(r'\((\d+)\)$')
    match = pattern.search(single_string)
    if match:
        number = int(match.group(1))
        modified_string = single_string[:match.start()].strip()
        return number, modified_string
    else: return None, single_string

def extract_time_and_remove(text):
    time_pattern = re.compile(r'\b(\d{1,2}:\d{2}[apAP][mM])\b')
    match = time_pattern.search(text)
    if match:
        extracted_time = match.group(1)
        modified_text = time_pattern.sub('', text).strip()
        return extracted_time, modified_text
    else: return None, text
