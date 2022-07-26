import json
import numpy as np
from config import *

import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pytorch_transformers import BertModel
import torch
from keybert import KeyBERT
from torch.autograd import Variable
from torch_geometric.data import HeteroData, Data

def load_raw_data_from_graph(save=True):
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    raw_data = list()
    cypher = "match (a1:upperclass)-[]-(m1)-[]-(paper)-[]-(n1)-[]-(a3:upperclass) where n1.upperclass = 'product' " \
             "and m1.upperclass = 'material' with paper.name as paper_name,a1, a3,m1 " \
             "match (a2:upperclass)-[]-(m2)-[]-(paper)-[]-(n2:`Faradaic efficiency`) " \
             "where paper.name=paper_name and m2.upperclass='method' " \
             "return a1.name as material_type, a2.name as method_type, " \
             "m2.name as method, m1.name as material, a3.name as product,n2.name as FE, paper.title as title, " \
             "paper.abstract as abstract"
    result = graph.run(cypher).to_data_frame()
    for _, value in result.iterrows():
        material = value['material']
        material_type = value['material_type']
        product = value['product']
        method = value['method']

        method_type = value['method_type']
        title = value['title']
        abstract = value['abstract']
        massage = abstract + title
        method_sentences = ' '.join([i for i in massage.split('.') if i.find(method) != -1])

        # location = [m.start() for m in re.finditer(method.replace('-', '').replace('+', ''), massage.replace('-', '').replace('+', ''))]
        #
        # method_sentences = ' '.join([massage[max(0, i-50):min(i+50, len(massage))] for i in location])

        faradaic_efficiency = re.findall(r'\d+(?:\.\d{1,3})?%', value['FE'].replace(' ', '').replace('+/-',''))
        if not faradaic_efficiency:
            faradaic_efficiency = re.findall(r'\d+(?:\.\d{1,3})?%?', value['FE'].replace(' ', '').replace('+/-',''))

        if faradaic_efficiency:
            faradaic_efficiency = min(100, max([float(i.replace('%', '')[:3]) for i in faradaic_efficiency]))
            # key_word = [i[0] for i in model.extract_keywords(abstract,keyphrase_ngram_range=(1,1))]
            data_line = {'material': material,
                         'product': product,
                         'method': method,
                         'method_type': method_type,
                         'material_type': material_type,
                         'label': faradaic_efficiency,
                         'keyword': method_sentences,
                         'title': title}
            raw_data.append(json.dumps(data_line))
            print(value['FE'], faradaic_efficiency)
        else:
            print('error'+str(value['FE']))

    if save:
        np.save('./data/raw_data3.npy', np.array(raw_data))
    return raw_data

def query_data():
    cypher = "match (a1:upperclass)-[]-(m1)-[]-(paper)-[]-(n1)-[]-(a3:upperclass) where a3.name = 'Cu-M' " \
             "and a1.name = 'CO' with paper.name as paper_name,a1, a3,m1 " \
             "match (a2:upperclass)-[]-(m2)-[]-(paper)-[]-(n2:`Faradaic efficiency`) " \
             "where paper.name=paper_name and a2.name='atomic level dispersion' " \
             "return a1.name as material_type, a2.name as method_type, " \
             "m2.name as method, m1.name as material, a3.name as product,n2.name as FE, paper.title as title, " \
             "paper.abstract as abstract"
    result = graph.run(cypher).to_data_frame()
    print(result)


def load_raw_data_from_graph_2(save=True):
    raw_data = list()
    cypher ="match (a1:upperclass)-[]-(m1)-[]-(paper)-[]-(n1) where n1.upperclass = 'product' " \
             "and m1.upperclass = 'material' with paper.name as paper_name,a1, n1,m1 " \
             "match (a2:upperclass)-[]-(m2)-[]-(paper)" \
             "where paper.name=paper_name and m2.upperclass='method' " \
             "return a1.name as material_type, a2.name as method_type, " \
             "m2.name as method, m1.name as material, n1.name as product, paper.title as title"
    result = graph.run(cypher).to_data_frame()
    for _, value in result.iterrows():
        material = value['material']
        material_type = value['material_type']
        product = value['product']
        title = value['title']
        method = value['method']
        method_type = value['method_type']
        data_line = {'material': material,
                     'product': product,
                     'material_type': material_type,
                     'title': title,
                     'method':method,
                     'method_type':method_type}
        raw_data.append(json.dumps(data_line))
    if save:
        np.save('./data/raw_data_all.npy', np.array(raw_data))
    return raw_data

def load_data_from_npy(data_file_name):
    return np.load(data_file_name).tolist()

def _preprocess_sample(sample_str):
    """
    preprocess each sample with the limitation of maximum length and pad each sample to maximum length
    :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
    :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
    """
    bert_model = BertModel.from_pretrained(model_name, cache_dir="./")
    raw_sample = json.loads(sample_str)
    a_str = ''
    sample = [[], []]
    for k in raw_sample.keys():
        if k == 'label':
            sample[1].append(raw_sample[k]/100)
        else:
            a_str += raw_sample[k]
            a_str += ' '
    token_ = tokenizer(a_str)['input_ids']

    sample[0] = bert_model(torch.tensor(token_).unsqueeze(0))[1].tolist()[0]
    return sample

def generate_bert_file():
    corpus_file = load_data_from_npy(raw_data_path + '/raw_data.npy')
    token_file = [_preprocess_sample(i) for i in corpus_file]
    np.save('./data/token_file.npy', np.array(token_file))

def Graph_Data():
    raw_data = np.load('./data/raw_data.npy').tolist()
    material_type = np.array([eval(i)['material_type'] for i in raw_data])
    data_type_encoder = LabelEncoder()
    data_type_encoded = data_type_encoder.fit_transform(np.unique(material_type))
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = data_type_encoded.reshape(len(data_type_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    method_type = np.array([eval(i)['method_type'] for i in raw_data])
    data_type_encoded = data_type_encoder.fit_transform(np.unique(method_type))
    integer_encoded = data_type_encoded.reshape(len(data_type_encoded), 1)
    onehot_encoded_2 = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded_material_type = np.hstack([onehot_encoded, np.zeros([onehot_encoded.shape[0], onehot_encoded_2.shape[1]])])
    onehot_encoded_method_type = np.hstack(
        [np.zeros([onehot_encoded_2.shape[0], onehot_encoded.shape[1]]), onehot_encoded_2])
    # method_type_x = [tokenizer.encode(i, max_length=20, pad_to_max_length=True) for i in np.unique(method_type)]
    x = np.vstack([onehot_encoded_material_type, onehot_encoded_method_type])

    material_type_mapping = {index_id: int(i) + 0 for i, index_id in enumerate(np.unique(material_type))}
    method_type_mapping = {index_id: int(i) + len(material_type_mapping) for i, index_id in enumerate(np.unique(method_type))}
    material_type_nodes = [material_type_mapping[index] for index in material_type]
    method_type_nodes = [method_type_mapping[index] for index in method_type]
    edge_index = torch.tensor([material_type_nodes, method_type_nodes])
    rev_edge_index = torch.tensor([method_type_nodes, material_type_nodes])
    data = Data()
    data.num_nodes = len(method_type_mapping) + len(material_type_mapping)
    data.edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
    # data.x = torch.ones((data.num_nodes, 20))
    data.x = torch.tensor(x, dtype=torch.float32)
    return data, material_type_mapping, method_type_mapping







if __name__ == '__main__':
    # data, material_type_mapping, method_type_mapping = Graph_Data()
    #query_data()
    raw_data = load_data_from_npy("./data/raw_data3.npy")
    import pandas as pd
    tb = pd.DataFrame()
    for i in raw_data:
        tb = tb.append(eval(i),ignore_index=True)
    tb.to_csv('raw_data3.csv')
    print(raw_data)
