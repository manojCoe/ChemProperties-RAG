import sys
import re

# Set default encoding to UTF-8
# sys.stdout.reconfigure(encoding='utf-8')

# Function to dynamically standardize and convert units
def convert_units_dynamic(text):
    unit_patterns = {
        r'(\w+)[ ]?g\s*-1': r'\1/g',         # Convert `mAhg^-1` or `mAhg -1` → `mAh/g`
        r'(\w+)[ ]?g\s*1': r'\1/g',          # Convert `mAhg 1` → `mAh/g`
        r'(\w+)[ ]?g\s*(\d+)': r'\1*g^\2',   # Convert `mAhg 2` → `mAh*g^2`
        r'(\w+)[ ]?kg\s*-1': r'\1/kg',       # Convert `Whkg^-1` → `Wh/kg`
        r'(\w+)[ ]?cm\s*-2': r'\1/cm²',      # Convert `mAcm^-2` → `mA/cm²`
        r'(\w+)[ ]?s\s*-1': r'\1/s',         # Convert `mV⋅s^-1` → `mV/s`
    }

    for pattern, replacement in unit_patterns.items():
        text = re.sub(pattern, replacement, str(text))

    return text
