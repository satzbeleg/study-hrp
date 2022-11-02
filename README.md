# study-hrp
Rediscovering Hashed Random Projections for efficient Quantization of Contextualized Sentence Embeddings.


### Install MiniConda for GPU
Please note that using GPUs for `tensorflow.sparse` is not adding speed improvements because most `tensorflow.sparse` functions only support CPU, i.e., the program would switch between GPU and CPU memory all the time what is very slow.

TensorFlow needs the CUDA drivers that available as Python packages only via Conda (Nvidia does not maintain PyPi packages).

```sh
conda install pip
conda create --name gpu-venv-study-hrp python=3.9 pip
conda activate gpu-venv-study-hrp
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install -r requirements.txt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
```

Install MiniConda if not exists
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# prevent conda autostart in shell
# conda config --set auto_activate_base false
rm Miniconda3-latest-Linux-x86_64.sh
```

Monitor
```
watch -n 0.5 nvidia-smi
```


### Download Datasets

```sh
# wget https://raw.githubusercontent.com/ulf1/sentence-embedding-evaluation-german/main/download-datasets.sh -O download-seeg.sh
mkdir datasets
nohup bash download-seeg.sh &
```

```sh
# wget https://raw.githubusercontent.com/facebookresearch/SentEval/main/data/downstream/get_transfer_data.bash -O download-senteval.sh
mkdir datasets-senteval
cp download-senteval.sh datasets-senteval/download-senteval.sh
cd datasets-senteval/ 
nohup bash download-senteval.sh &
cd ..
```

### Download Models

```sh
nohup bash download-sbert.sh &
nohup bash download-tfhub.sh &
python -m laserembeddings download-models
```

### Run all experiments
```sh
nohup bash experiments.sh &
```

