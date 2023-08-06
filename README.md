## FROMAGe

### Setup

1. Create conda environment: `conda env create -f environment.yml`

2. Activate conda environment: `conda activate fromage`

3. Install packages via pip: `pip install -r requirements.txt`

4. Download `MIMIC_JPG_<SPLIT>.tsv`files and `biovil_backbone_2048.pt` from [Drive](https://drive.google.com/drive/u/0/folders/1w-JpJGtBCEgXpAdbm4qmdvE8EGjo-KkZ)

5. Place `MIMIC_JPG_<SPLIT>.tsv files` under `data/` folder and `biovil_backbone_2048.pt` under `bin/` folder.

6. Run `pip install -e .` to install the fromage as an editable package.

### Run Experiments

To run experiments, you need to login to Weights & Biases from your terminal ([Link](https://docs.wandb.ai/quickstart#2-log-in-to-wb), select the command line option).

For this project, we follow YAML-based configuration. You need to change `logger/name` and `logger/version` for every run that you create. The model reads configuration from `config/train.yaml` file.

After logging in to Weights & Biases and setting config, you can run `fromage_train.sh` script by `sbatch fromage_train.sh` command.

For any questions, please get in contact with [Osman Batur](mailto:osmanbaturince@gmail.com).
