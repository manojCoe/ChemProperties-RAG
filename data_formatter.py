import json

# Raw JSON string with incorrectly displayed Unicode
json_data = """
{
    "electrolyte_composition": "LiCl solution with addition of betaine",
    "coulombic_efficiency": "maintained after 500 cycles with a high capacity retention rate of 92 %",
    "pH_values": "N/A",
    "salt_concentration": "2 M LiCl (1.6 g betaine)",
    "operating_voltage": "up to 1.8 V",
    "electrochemical_stability": "Electrolyte structure with an expanded electro-chemical stability window of 2.91 V",
    "specific_capacity": "60.4 mAh g\u00b11",
    "energy_density": "69 Wh kg\u00b11 and 55 Wh kg\u00b11",
    "capacity_retention": "92 %",
    "viscosity": "N/A"
}
"""

# Load JSON and ensure correct encoding
decoded_json = json.loads(json_data)

# Ensure all strings are properly decoded
for key, value in decoded_json.items():
    if isinstance(value, str):
        decoded_json[key] = value.encode('utf-8').decode('utf-8')

# Print the properly formatted JSON
print(json.dumps(decoded_json, indent=4, ensure_ascii=False))
