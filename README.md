# MODAPro
MODAPro: Heterogeneous Graph convolutional networks framework for Interpretable Mining novel Hub Molecules with Functional Prompt
1. Code Runtime Environment
【Packaging the Virtual Environment】

2. Preparing Raw Omics Data
Requirements:
   1) Feature Information: Prepare a profile for each omics type, with rows as molecules and columns as sample numbers, including a second column for omics type. For single omics data input, the second column must still indicate the omics type.
   2) Sample Grouping Information: The first column should be labeled as "Sample" and match the sample numbers in the feature information; the second column should be labeled as "group" and contain sample grouping information.
   
3. Running Data Preprocessing Procedures
   1) Data is stored in the data/ directory. Name the project: data/TCGA_GBM.
   2) Fill in the parameter file: data/Params.txt.
   3) Run the script to convert raw profiles into machine learning matrices: data_preprocess/Cal_ML_Index.py.
   4) Convert input molecule IDs: data_preprocess/id_trans.py.
   5) Extract disease-specific networks from the background network: data_preprocess/Tidy_biological_network.py.

4. Executing MODAPro
   1) Fill in the parameter file: Params.txt.
   2) Execute the main function: main.py.
