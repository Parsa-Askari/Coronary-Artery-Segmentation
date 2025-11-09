import os 
import json
import nbformat as nbf
from nbconvert import PythonExporter
from tqdm import tqdm
from pathlib import Path
import re
settings={
    "scripts_paths":[
        "./utils/dataset.py",
        "./utils/helpers.py",
        "./utils/logger.py",
        "./utils/preprocessing.py",
        "./utils/recorder.py",
        "./models/nnunet_blocks.py",
        "./models/nnunet.py",
        "./utils/losses.py",
        "./trainer.py"
    ],
    "notebook_paths":[
        "./nnUnetAttention.ipynb"
    ]
}

imports_list = []
codes_list = []
cells = []
def process_scripts(paths,imports_list,codes_list):
    temp = set()
    for path in tqdm(paths):
        print(f"processing {path}")
        text =  Path(path).read_text()
        lines = text.splitlines()
        i=0
        
        while lines[i].strip() != "###IE###" : 
            temp.add(lines[i].strip())
            i+=1
        while lines[i].strip() != "###SS###" :
            i+=1

        i+=1
        code_bocks = lines[i:]
        title = (path.split("/")[-1]).split(".")[0]
        codes_list.append(nbf.v4.new_markdown_cell(f"# {title}"))
    
        code_string = "\n".join(code_bocks)
        codes_list.append(nbf.v4.new_code_cell(code_string))


    temp = list(temp)
    temp = "\n".join(temp)
    imports_list.append(nbf.v4.new_code_cell(temp))
    
    return imports_list , codes_list
nb = nbf.v4.new_notebook()
imports_list , codes_list = process_scripts(
    settings["scripts_paths"],
    imports_list,
    codes_list,
)
cells = list(imports_list) + codes_list
nb.cells = cells
with open("merged.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
