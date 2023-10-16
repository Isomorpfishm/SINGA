![SINGA banner and concept](img/SINGA.png)

## SINGA - Molecular <ins>S</ins>ampling with Protein-Ligand <ins>IN</ins>teractions aware <ins>G</ins>enerative <ins>A</ins>dversarial Network

### This project is under progress. Coming soon.

Developer: OON Yu Yang (翁宇陽), Project Officer (Computational Biology || Biochemistry) at School of Biological Sciences, Nanyang Technological University (NTU), Singapore

Principal Investigator: **[Assoc Prof MU Yuguang](https://dr.ntu.edu.sg/cris/rp/rp00074?ST_EMAILID=YGMU)**

## Introduction

![Protein-ligand complex as three-dimensional heterogeneous graph](img/PLGraph.png)
<p align="center">
    <em>Protein-ligand complex as a three-dimensional heterogeneous graph, with an emphasis on interactions between the protein and ligand.</em>
</p>

## Installation

Working inside a Conda virtual environment is hightly encouraged, but not necessary. Create a new environment with the command:

```
conda env create -f environment.yml
```

Otherwise, install the following packages and dependencies sequentially:

```
conda create -n SINGA python=3.10
conda activate SINGA
conda install -c conda-forge openbabel
conda install mkl-service
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.1 torch-scatter==2.1.1 torch-sparse==0.6.17 torch-cluster==1.6.3
pip install biopandas==0.4.1 pytorch-lightning==2.0.6
pip install rdkit==2023.3.3 oddt==0.7
pip install e3nn==0.5.1 dgl==1.1.2
pip install tensorboard==2.14.1
pip install pybel==0.15.5
pip install termcolor==2.3.0 easydict==1.10
```

## Directory tree

```
  $ singa (main directory)
  .
  |
  |__ /autodock_vina
  |__ /ckpt
  |__ /config
  |__ /dataset
      |__ /crossdocked_graph10_v3
  |__ /example
  |__ /img
  |__ /logs
  |__ /model
      |__ __init__.py
      |__ BeamSearch.py
      |__ CProMG.py
      |__ EF_layers.py
      |__ Embedding.py
      |__ GAN.py
      |__ Jd.pt
  |__ /output
  |__ /utils
      |__ /ledock
      |__ __init__.py
      |__ Data.py
      |__ Featuriser.py 
      |__ fpscores.pkl.gz
      |__ gen.py
      |__ misc.py
      |__ PLFeature.py
      |__ PLIExtension.py
      |__ PLInteraction.py
      |__ PLParser.py
      |__ redirect.py
      |__ SAScorer.py
      |__ Stopper.py
  |__ .gitignore
  |__ __init__.py
  |__ environment.yml
  |__ LICENSE
  |__ MakeGraph.py
  |__ README.md
  |__ gen.py
  |__ train.py
```

## License

MIT License

## Acknowledgement

Part of this codebase is adapted from [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2) and [CProMG](https://github.com/lijianing0902/CProMG). Details of the adaptation are stated explicitly in the script.
