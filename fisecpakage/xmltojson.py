import xmltodict
import json
import sys

def xml_string_to_json(xml_string):
    try:
        print(xml_string)
        sys.setrecursionlimit(2000)
        # Parse XML string to dict
        dict_data = xmltodict.parse(xml_string, item_depth=10)
        
        # Convert dict to JSON
        # json_data = json.dumps(dict_data, indent=2)
        return dict_data
        
    except Exception as e:
        print(f"Error converting XML to JSON: {str(e)}")
        return None

