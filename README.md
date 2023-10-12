# STACCI
# Installation Instructions

1. Clone the repository to your local machine and enter the repository in the command line interface.
2. Use conda to create a new environment according to environment.yml

   `conda env create -f environment.yml`

   The purpose of this step is to install python, cudatoolkit and cudnn, where the versions of cudatoolkit and cudnn must correspond. The version in the .yml file is applicable to hosts with cuda ≥ 11.3. For servers with cuda lower than this version, consider upgrading cuda or finding the corresponding cudatoolkit version and cudnn version.

   Specifying the python version is to facilitate the next step to find the corresponding version of torch_geometric related packages.
3. In the new environment, install the specified version of pytorch and torch_geometric related packages

   **Don't forget to activate the env**

   - First install pytorch related packages

     `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`

   The version of pytorch should be suitable for the version of cudatoolkit. The above command is from the pytorch official website and is the latest version that cuda 11.3 can install.

   Those with different cuda versions can find the appropriate command on [this website](https://pytorch.org/get-started/previous-versions/).

   - Then install torch_geometric related packages

   There are five torch_geometric related packages: torch_spline_conv, torch_sparse, torch_scatter, torch_cluster, pyg_lib, of which pyg_lib should be installed last.

   The version of the above packages is related to the system architecture, operating system, python version, cuda version and torch version. If the package version of each step is consistent with the tutorial, you can directly download the wheel file in the following link for installation:

   Link: [https://pan.baidu.com/s/1FqA9KFENfk4RSOMLmblyiw](https://pan.baidu.com/s/1FqA9KFENfk4RSOMLmblyiw) Password: 8rvh

   `pip install torch_spline_conv-1.2.1+pt112cu113-cp310-cp310-linux_x86_64.whl`

   `pip install torch_sparse-0.6.16+pt112cu113-cp310-cp310-linux_x86_64.whl`

   `pip install torch_scatter-2.1.0+pt112cu113-cp310-cp310-linux_x86_64.whl`

   `pip install torch_cluster-1.6.0+pt112cu113-cp310-cp310-linux_x86_64.whl`

   `pip install pyg_lib-0.3.0+pt112cu113-cp310-cp310-linux_x86_64.whl`

   Otherwise, please download the appropriate wheel file from [this website](https://data.pyg.org/whl/), and note that the above installation commands should also be modified accordingly.

   Finally, install torch_geometric:

   `pip install torch_geometric`
4. `pip install STACCI`

# Usage Instructions

After creating a new environment according to the installation instructions and installing the corresponding dependencies, place the .h5ad file of the dataset in the specified file structure, specifically, the desired file structure of the dataset is as follows:

```bash
{root}
└── {dataset_path}
    └── {dataset}
        └── {h5_name}.h5ad
```

{x} represents the value of the variable x, and the four custom run result saving folders {generated_path}, {embedding_path}, {model_path}, {result_path} will be automatically created in the {root} folder.

After setting up the file structure, enter the STACCI/ folder under the repository and execute the following command:

```bash
python pipeline.py --root {root} --ds-dir {dataset_path} --ds-name {dataset} --h5-name {h5_name}
```

An example command is:

```bash
python pipeline.py --root ../tests/ --ds-dir datasets/ --ds-name T25_F1 --h5-name T25_F1_1000hvg_ceco
```

The complete file structure of the repository including the example dataset should be as follows:

```bash
STACCI
├── README.md
├── pyproject.toml
├── environment.yml
├── .gitignore
├── STACCI
│   ├── __init__.py
│   ├── data_handler.py
│   ├── model.py
│   ├── pipeline.py
│   ├── trainer.py
│   └── utils.py
└── tests
    └── datasets
        └── T25_F1
            └── T25_F1_1000hvg_ceco.h5ad
```
