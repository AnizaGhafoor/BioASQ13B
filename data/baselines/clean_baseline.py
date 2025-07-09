import os, json
from tqdm import tqdm

""" 

This script intends to simplify the downloaded baselines by removing unnecessary information and keeping only some essential key-value pairs (pmid, title, and abstract). Afterward, it updates the baseline file with the essential information only. 

"""

folder_path = "baselines/"

def filter(json_obj):
    return {key: json_obj[key] for key in ['pmid', 'title', 'abstract'] if key in json_obj}

for filename in os.listdir(folder_path):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(folder_path, filename)
        
        print(f"Processing the file '{file_path}'.")
        filtered_content = []
        with open(file_path, 'r') as file:
            for line in tqdm(file):
                content = json.loads(line)
                filtered = filter(content)
                filtered_content.append( json.dumps(filtered) )
            
        print(f"Writing the clean content in file '{file_path}'.")
        with open(file_path, 'w') as file:
            file.write('\n'.join(filtered_content))
            
        print(f">> File '{file_path}' has been processed.\n")

print("All files have been processed.")