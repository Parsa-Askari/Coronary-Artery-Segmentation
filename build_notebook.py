import os 
import json
import nbformat as nbf
from nbconvert import PythonExporter
from tqdm import tqdm
from pathlib import Path
import re
from ordered_set import OrderedSet


MAIN_GUARD_RE = re.compile(
    r'(?m)^[ \t]*if __name__ == [\'"]__main__[\'"]:\n(?:[ \t]+.*\n?)*'
)

def strip_main_guard(code: str) -> str:
    """Remove `if __name__ == "__main__":` blocks from a Python source string."""
    return MAIN_GUARD_RE.sub("", code)

def process_scripts(paths, imports_list, codes_list):
    temp = OrderedSet()
    for path in tqdm(paths):
        print(f"processing {path}")
        text = Path(path).read_text()
        lines = text.splitlines()
        i = 0

        while lines[i].strip() != "###IE###":
            temp.add(lines[i].strip())
            i += 1
        while lines[i].strip() != "###SS###":
            i += 1

        i += 1
        code_bocks = lines[i:]
        title = (path.split("/")[-1]).split(".")[0]
        codes_list.append(nbf.v4.new_markdown_cell(f"# {title}"))

        code_string = "\n".join(code_bocks)

        code_string = strip_main_guard(code_string).rstrip()

        codes_list.append(nbf.v4.new_code_cell(code_string))

    temp = list(temp)
    temp = "\n".join(temp)
    imports_list.append(nbf.v4.new_code_cell(temp))

    return imports_list, codes_list

def process_notebook(paths,imports_list,codes_list):
    for path in tqdm(paths):
        with open(path,"r") as f:

            nb = nbf.read(f, as_version=4)
        exporter = PythonExporter()
        source, _ = exporter.from_notebook_node(nb)
        with open("./temp_script.py", "w") as f:
            f.write(source)
        imports_list , codes_list = process_scripts(
            ["./temp_script.py"],
            imports_list=imports_list,
            codes_list=codes_list
        )
        os.remove("./temp_script.py")
    return imports_list , codes_list

def build_kaggle_project(outpath,notebook_name = "nnUnetAttention.ipynb"):
    imports_list = []
    codes_list = []
    cells = []
    KAGGLE_HEADER_METADATA = {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "version": "3.6.4",
            "file_extension": ".py",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "name": "python",
            "mimetype": "text/x-python",
        },
        "kaggle": {
            "accelerator": "none",
            "dataSources": [
                {
                    "sourceId": 13970329,
                    "sourceType": "datasetVersion",
                    "datasetId": 8906155,
                }
            ],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": False,
        },
    }
    nb = nbf.v4.new_notebook()
    nb.metadata.setdefault("kaggle", {})
    nb.metadata["kaggle"].update(KAGGLE_HEADER_METADATA)

    settings={
        "scripts_paths":[
            "./utils/dataset.py",
            "./utils/helpers.py",
            "./logger.py",
            "./utils/preprocessing.py",
            "./utils/recorder.py",
            "./models/costume_nnunet_blocks.py",
            "./models/nnunet.py",
            "./models/swin_encoder.py",
            "./models/swin_blocks.py",
            "./utils/losses.py",
            "./trainer.py"
        ],
        "notebook_paths":[
            f"./{notebook_name}"
        ]
    }
    imports_list , codes_list = process_scripts(
        settings["scripts_paths"],
        imports_list,
        codes_list,
    )


    imports_list , codes_list = process_notebook(
        settings["notebook_paths"],
        imports_list,
        codes_list
    )
    install_cell = nbf.v4.new_code_cell(
        "# Install required packages\n"
        "!pip install -q zarr segmentation-models-pytorch"
    )

    # Put install cell first
    cells = [install_cell] + list(imports_list) + codes_list

    nb.cells = cells
    path = os.path.join(outpath,"merged.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
if __name__ =="__main__":
    build_kaggle_project("./")