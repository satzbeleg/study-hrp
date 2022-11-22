import argparse
import logging
import tensorflow as tf
import sentence_transformers as sbert
import laserembeddings
import tensorflow_hub
import tensorflow_text 
import senteval
import sentence_embedding_evaluation_german as seeg
import os
import numpy as np
import json


# -----------------------------------------------
# parse input arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model", type=str, default=None)
parser.add_argument("--num-bool-features", type=int, default=None)
parser.add_argument("--random-state", type=int, default=None)
parser.add_argument("--output-type", type=str, default=None)
args = parser.parse_args()


# -----------------------------------------------
LOGFOLDER = os.path.join("results", args.model.replace('/', '_'))
os.makedirs(LOGFOLDER, exist_ok=True)

RESULTFILEPATH = (
    LOGFOLDER + "/"
    f"numbool={args.num_bool_features}-"
    f"randomstate={args.random_state}-"
    f"outputtype={args.output_type}")

# -----------------------------------------------
# logging settings
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"{RESULTFILEPATH}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)

# -----------------------------------------------
# (1a) Load pre-trained model
model_list_sbert = [
    'paraphrase-multilingual-mpnet-base-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'distiluse-base-multilingual-cased-v2',
    'sentence-transformers/LaBSE',
]

if args.model in model_list_sbert:
    model_embed = sbert.SentenceTransformer(args.model)
    logger.info(f"SBert model {args.model} is loaded.")
    tmp = model_embed.encode(["get dims."])
    NUM_FEATURES = tmp.shape[1]
    logger.info(f"Num of SBert features: {NUM_FEATURES}")

    def call_model_embed(sentences):
        return model_embed.encode(sentences)

elif args.model in ['laser-en', 'laser-de']:
    model_embed = laserembeddings.Laser()
    logger.info(f"Laser model is loaded.")
    tmp = model_embed.embed_sentences(["get dims."], lang=args.model[-2:])
    NUM_FEATURES = tmp.shape[1]
    logger.info(f"Num of Laser features: {NUM_FEATURES}")

    def call_model_embed(sentences):
        return model_embed.embed_sentences(sentences, lang=args.model[-2:])

elif args.model in ['m-use']:
    model_embed = tensorflow_hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    logger.info(f"m-USE model is loaded.")
    tmp = model_embed(["get dims."])
    NUM_FEATURES = tmp.shape[1]
    logger.info(f"Num of m-USE features: {NUM_FEATURES}")

    def call_model_embed(sentences):
        return model_embed(sentences).numpy()

# -----------------------------------------------
# (1b) Specify HRP layer
# see https://github.com/satzbeleg/evidence-model-v0.6x/blob/
#     66cd55eccfb435ea2f62fbfcce4e4a5ac9fa92dd/evidence_model/hrp.py#L8
class HashedRandomProjection(tf.keras.layers.Layer):
    def __init__(self,
                 hyperplane=None,
                 random_state=42,
                 output_size=None,
                 **kwargs):
        super(HashedRandomProjection, self).__init__(**kwargs)
        self.hyperplane = hyperplane
        self.random_state = random_state
        self.output_size = output_size
    def build(self, input_shape=None):
        if self.hyperplane is None:
            num_features = input_shape[-1]
            tf.random.set_seed(self.random_state)
            self.hyperplane = tf.Variable(
                initial_value=tf.random.normal(
                    shape=(num_features, self.output_size)),
                trainable=False)
        else:
            self.hyperplane = tf.Variable(
                initial_value=self.hyperplane,
                trainable=False, dtype=self.dtype)
        super(HashedRandomProjection, self).build(input_shape)
    def call(self, inputs):
        projection = tf.matmul(inputs, self.hyperplane)
        hashvalues = tf.experimental.numpy.heaviside(projection, 0)
        return hashvalues


if args.output_type == "hrp":
    # call lateron
    model_hrproj = HashedRandomProjection(
        output_size=args.num_bool_features,
        random_state=args.random_state
    )
    # build HRP layer
    model_hrproj.build(input_shape=(NUM_FEATURES,))


# -----------------------------------------------
# (2a) SentEval Preprocess Functions

# specify `prepare`
def senteval_prepare(params, samples):
    return

# specify `batcher`
# the `batch` contains a list of token lists
if args.output_type == "hrp":
    def senteval_preprocess(params, batch):
        sentences = [' '.join(s) for s in batch]
        features = call_model_embed(sentences)
        hashvalues = model_hrproj(tf.convert_to_tensor(features))
        return hashvalues.numpy()

elif args.output_type == "sigmoid":
    def senteval_preprocess(params, batch):
        sentences = [' '.join(s) for s in batch]
        features = call_model_embed(sentences)
        return (features > 0.0).astype(np.float32)  # rounded sigmoid 

elif args.output_type == "float":
    def senteval_preprocess(params, batch):
        sentences = [' '.join(s) for s in batch]
        features = call_model_embed(sentences)
        return features

# -----------------------------------------------
# (3a) SentEval settings

# p.3 in https://arxiv.org/pdf/1803.05449.pdf
# senteval_params = {
#     'task_path': './', 
#     'usepytorch': True,
#     'kfold': 10,
#     'classifier': {
#         'nhid': 0, 'optim': 'adam', 'batch_size': 64,
#         'tenacity': 5, 'epoch_size': 4}
# }
# https://github.com/facebookresearch/SentEval#senteval-parameters
senteval_params = {
    'task_path': './', 
    'usepytorch': True,
    'kfold': 5,
    'classifier': {
        'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
        'tenacity': 3, 'epoch_size': 2}
}


# -----------------------------------------------
# (4a) SentEval downstream tasks

# p.6, https://arxiv.org/pdf/1908.10084.pdf 
senteval_tasks = [
    'MR', 'CR', 'SUBJ', 'MPQA', 'SST5', 'TREC', 'MRPC',
    'STS12', 'STS13', 'STS14', 'STS15', 'STS16']


# -----------------------------------------------
# (5a) Run SentEval

se = senteval.engine.SE(senteval_params, senteval_preprocess, senteval_prepare)
senteval_results = se.eval(senteval_tasks)

with open(f"{RESULTFILEPATH}-senteval.json", 'w') as fp:
    json.dump(senteval_results, fp)


# -----------------------------------------------
# (2b) SEEG Preprocess Functions

# the `batch` contains a list of strings
if args.output_type == "hrp":
    def seeg_preprocess(sentences):
        features = call_model_embed(sentences)
        hashvalues = model_hrproj(tf.convert_to_tensor(features))
        return hashvalues.numpy()

elif args.output_type == "sigmoid":
    def seeg_preprocess(sentences):
        features = call_model_embed(sentences)
        return (features > 0.0).astype(np.float32)  # rounded sigmoid 

elif args.output_type == "float":
    def seeg_preprocess(sentences):
        features = call_model_embed(sentences)
        return features

# -----------------------------------------------
# (3b) SEEG settings

seeg_params = {
    'datafolder': './datasets',
    'bias': True,
    'balanced': True,
    'batch_size': 128, 
    'num_epochs': 500,
}

# -----------------------------------------------
# (4b) SEEG downstream tasks

seeg_tasks = ['VMWE', 'ABSD-2', 'MIO-P', 'ARCHI', 'LSDC']

# -----------------------------------------------
# (5b) Run SEEG 

seeg_results = seeg.evaluate(seeg_tasks, seeg_preprocess, **seeg_params)

with open(f"{RESULTFILEPATH}-seeg.json", 'w') as fp:
    json.dump(seeg_results, fp)
