# MODAPro
MODAPro: Explainable Heterogeneous Networks with Variational Graph Autoencoder for Mining Disease-Specific Functional Molecules and Pathways from Omics Data
License: MODAPro is distributed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0) for all original code. The software also incorporates third-party components released under permissive open-source licenses (MIT, Apache 2.0, BSD), which retain their original license terms.

1. Code Runtime Environment: conda env create -f path\environment.yml
2. Installed 'sparse_tools', referred to: https://github.com/Yangxc13/sparse_tools
   
3. Preparing Raw Omics Data
Requirements:
   1) Feature Information: Prepare a profile for each omics type, with rows as molecules and columns as sample numbers, including a second column for omics type. For single omics data input, the second column must still indicate the omics type.
   2) Sample Grouping Information: The first column should be labeled as "Sample" and match the sample numbers in the feature information; the second column should be labeled as "group" and contain sample grouping information.
   
4. Running Data Preprocessing Procedures
   1) Data is stored in the data/ directory. Name the project: data/TCGA_GBM.
   2) Fill in the parameter file: data/Params.txt.
   3) Run the script to convert raw profiles into machine learning matrices: data_preprocess/Cal_ML_Index.py.
   4) Convert input molecule IDs: data_preprocess/id_trans.py.
   5) Extract disease-specific networks from the background network: data_preprocess/Tidy_biological_network.py.

5. Executing MODAPro
   1) Fill in the parameter file: Params.txt.
   2) Execute the main function: main.py.

6. Downloaded example data
   1) Background biological network data: Files shared via web disk: biological_network
Link: https://pan.baidu.com/s/1975Kx6vgzeDTIE2f4XxNig Extract code: 0714. After downloading files and saving at 'database/biological_network'.
   2) Example inputted data: Files shared via web disk: TCGA_GBM
Link: https://pan.baidu.com/s/1difKOcxmVCU7lVkyIt43eA Extract code: 0714. After downloading files and saving at 'data/TCGA_GBM'.
