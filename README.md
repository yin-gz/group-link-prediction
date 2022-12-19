## Beyond Individuals: Modeling Mutual and Multiple Interactions for Inductive Link Prediction between Groups

### Dependencies
- Compatible with PyTorch 1.8 and Python 3.8.
- Dependencies can be installed using `requirements.txt`.

### Preparation
- Make directories: `./checkpoints`, `./log`, `./wandb`

### Datasets
- We conduct experiments on three datasets: MAG-G, Aminer-G, and Weeplaces-G. The datasets can be downloaded [here](https://drive.google.com/drive/folders/1UWTdSy6L4aVI8zIHlzTywM592LdFaI3T?usp=sharing)
- The datasets should be put in `./data` directory.

### Training and Evaluating a model:
- Params for reproducing the reported GE+MMAN results are saved in `./config`. Please run
```shell
    #### Method: GE+MMAN
    # Aminer-G
    python run_academic_group.py -config_file Aminer-G.yml
    
    # MAG-G
    python run_academic_group.py -config_file MAG-G.yml
    
    # Weeplaces-G
    python run_weeplaces.py -config_file Weeplaces-G.yml
```
- Commands for reproducing the other methods (take MAG-G dataset as a example):

```shell
  # -----------Plain GNN-based Encoders----------- #
  # GCN
  python run_academic_group.py -dataset MAG-G -only_GE True -graph_based GraphSage -score_method MLP
  
  # -----------GE with Aggregating Methods----------- #
  # GE-AVG
  python run_academic_group.py -dataset MAG-G -only_GE False -graph_based GraphSage -i2g_method avg -score_method MLP
  
  # GE-MMAN
  python run_academic_group.py -dataset MAG-G -only_GE False -graph_based GraphSage -i2g_method MMAN -view_num 4 -score_method mv_score
```

  - `-only_GE` denotes only using the GNN-based Encoders without aggregators.
  - `-graph_based` denotes GNN encoder type.
  - `-i2g_method` denotes the aggragating method. It can take the following values:
    - `avg`/`degree`/`att`/`set2set`/`MMAN`/`MAB`
  - `-score_method` is some name given for the run (used for storing model parameters)
    - `mv_score` for `MMAN`
    - `MLP` for other aggragating methods
  - Rest of the arguments can be listed using `python run_academic_group.py -h` or `python run_weeplaces.py -h`

### Citation:
Please cite the following paper if you use this code in your work.
```bibtex
@inproceedings{
    Yin2023beyondindividuals,
    title={Beyond Individuals: Modeling Mutual and Multiple Interactions for Inductive Link Prediction between Groups},
    author={Gongzhu Yin and Xing Wang and Hongli Zhang and Chao Meng and Yuchen Yang and Ku Lu and Yi Luo},
    booktitle={ACM International Conference on Web Search and Data Mining},
    year={2023}
}
```
For any questions or suggestions please contact [Gongzhu Yin](yingz@hit.edu.cn).