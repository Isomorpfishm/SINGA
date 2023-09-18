![SINGA banner and concept](img/SINGA.png)

## SINGA - Molecular <ins>S</ins>ampling with Protein-Ligand <ins>IN</ins>teractions aware <ins>G</ins>enerative <ins>A</ins>dversarial Network

### This project is under progress. Coming soon.

Developer: OON Yu Yang (翁宇陽), Project Officer (Computational Biology || Biochemistry) at School of Biological Sciences, Nanyang Technological University (NTU), Singapore

Principal Investigator: **[Assoc Prof MU Yuguang](https://dr.ntu.edu.sg/cris/rp/rp00074?ST_EMAILID=YGMU)**

## Introduction

![Protein-ligand complex as three-dimensional heterogeneous graph](img/PLComplex.png)
<p align="center">
    *Protein-ligand complex as a three-dimensional heterogeneous graph, with an emphasis on protein-ligand interactions.*
</p>

## Directory tree

```
  $ singa (main directory)
  .
  |
  |-- /autodock_vina
  |-- /ckpt
  |-- /config
  |-- /dataset
      |__ /crossdocked_graph10
  |-- /example
  |-- /features
  |-- /img
  |-- /model
      |-- Discriminator.py
      |-- EF_embedding.py
      |-- EF_layers.py
      |-- GAN.py
      |-- Generator.py
      |__ Masking.py
  |-- /output
  |-- /utils
      |-- /ledock
      |-- __init__.py
      |-- Data.py
      |-- Featuriser.py 
      |-- misc.py
      |-- PLFeature.py
      |-- PLIExtension.py
      |-- PLInteraction.py
      |-- PLParser.py
      |__ redirect.py
  |-- .gitignore
  |-- __init__.py
  |-- environment.yml
  |-- LICENSE
  |-- MakeGraph.py
  |-- README.md
  |__ train.py
```

## License

MIT License

## Acknowledgement

Part of this codebase is adapted from [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2) and [HGScore](https://github.com/KevinCrp/HGScore). Details of the adaptation are stated explicitly in the script.
