import pyterrier as pt
if not pt.started():
  pt.init()
from pyterrier_pisa import PisaIndex
import json, os, glob


def create_index(file_path):
    # get the folder name
    folder_name = file_path.split('/')[-1].split(".")[0].split("_")
    folder_name.pop(0)
    folder_name = "_".join(folder_name)
        
    index_path = "../../data/indexes/" + folder_name
    
    if not os.path.exists(index_path):
        os.makedirs(index_path)
        print(f"Folder '{index_path}' has been created.")

    return PisaIndex(index_path, text_field='text')


def load_collection(doc_path):
    latest_docs = {} 
    
    with open(doc_path, "r") as f:
        for pub in map(json.loads, f):
            # replace the entry for a pmid if it already exists
            latest_docs[pub["pmid"]] = {
                "docno": pub["pmid"],
                "text": " ".join([pub["title"], pub["abstract"]])
            }
    
    for doc in latest_docs.values():
        yield doc
            

def main():
    """ main """
    folder_path = "../../data/baselines"
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))

    for file_path in jsonl_files:
        
        index = create_index(file_path)
        print(f"Pisa index created in '{index.path}'.")
        
        index.index( load_collection(file_path) )
        print(f"Collection '{file_path}' has been indexed successfully.")
        

if __name__ == "__main__":
    main()
