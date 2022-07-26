from transformers import AutoTokenizer
from py2neo import Graph
from transformers import BertTokenizer
neo4j_url = 'http://10.0.82.91:7474'
user = 'neo4j'
pwd = 'bigdata'
graph = Graph(neo4j_url, auth=(user, pwd))

# checkpoint_token = 'allenai/scibert_scivocab_uncased'
checkpoint_token = './bert-base-uncased'
# model_name = '/data/AIplusMaterials/pretrain_bert_with_maskLM-main/checkpoint/epoch_2/'
tokenizer = AutoTokenizer.from_pretrained(checkpoint_token)
# tokenizer = BertTokenizer(vocab_file='./pretrained_models/vocab.txt')
model_name = './bert-base-uncased'
raw_data_path = './data'
str_index = ['material', 'product', 'method', 'method_type', 'material_type', 'title', 'keyword', 'label']
str_max_len = [30, 5, 30, 5, 5, 30, 80]


accumulation_steps = 4
epoch = 400



