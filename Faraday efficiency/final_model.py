import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertModel as BM
from transformers import BertModel as tBert
import torch
import config
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class model_1(nn.Module):
    """
    只有BERT
    """
    def __init__(self):
        super(model_1, self).__init__()
        self.model_material = BM.from_pretrained(config.model_name)
        self.model_product = BM.from_pretrained(config.model_name)
        self.model_method = BM.from_pretrained(config.model_name)
        self.model_method_type = BM.from_pretrained(config.model_name)
        self.model_material_type = BM.from_pretrained(config.model_name)
        self.model_title = BM.from_pretrained(config.model_name)
        self.key_word = BM.from_pretrained(config.model_name)
        self.linear1 = nn.Linear(768*5, 800)
        self.linear2 = nn.Linear(800, 200)
        self.linear3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _process_data(self, data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)
        output = model(data, attention_mask=mask)[1].reshape(-1, 768)
        return output

    def forward(self, material, product, method, method_type, material_type, title, keyword, graph_embed):
        material = self._process_data(material, self.model_material)
        product = self._process_data(product, self.model_product)
        method = self._process_data(method, self.model_method)
        title = self._process_data(title, self.model_title)
        keyword = self._process_data(keyword, self.key_word)
        out = torch.cat([material, product, keyword, method, title], 1)
        total = self.linear1(out)
        out = self.relu(total)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out

class model_2(nn.Module):
    """
    BERT + Type Embedding
    """
    def __init__(self):
        super(model_2, self).__init__()
        self.model_material = BM.from_pretrained(config.model_name)
        self.model_product = BM.from_pretrained(config.model_name)
        self.model_method = BM.from_pretrained(config.model_name)
        self.model_method_type = BM.from_pretrained(config.model_name)
        self.model_material_type = BM.from_pretrained(config.model_name)
        self.model_title = BM.from_pretrained(config.model_name)
        self.key_word = BM.from_pretrained(config.model_name)
        self.linear_type = nn.Linear(23, 8)
        self.linear1 = nn.Linear(768*5 + 8, 800)
        self.linear2 = nn.Linear(800, 200)
        self.linear3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _process_data(self, data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)
        output = model(data, attention_mask=mask)[1].reshape(-1, 768)
        return output

    def forward(self, material, product, method, method_type, material_type, title, keyword, graph_embed):
        material = self._process_data(material, self.model_material)
        product = self._process_data(product, self.model_product)
        method = self._process_data(method, self.model_method)
        title = self._process_data(title, self.model_title)
        keyword = self._process_data(keyword, self.key_word)
        type = self.relu(self.linear_type(torch.cat([method_type, material_type], 1)))
        out = torch.cat([material, product, keyword, method, title, type], 1)
        total = self.linear1(out)
        out = self.relu(total)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out

class model_3(nn.Module):
    """
    BERT + Graph
    """
    def __init__(self):
        super(model_3, self).__init__()
        self.model_material = BM.from_pretrained(config.model_name)
        self.model_product = BM.from_pretrained(config.model_name)
        self.model_method = BM.from_pretrained(config.model_name)
        self.model_method_type = BM.from_pretrained(config.model_name)
        self.model_material_type = BM.from_pretrained(config.model_name)
        self.model_title = BM.from_pretrained(config.model_name)
        self.key_word = BM.from_pretrained(config.model_name)
        self.linear1 = nn.Linear(768*5 + 2*5, 800)
        self.linear2 = nn.Linear(800, 200)
        self.linear3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def _process_data(self, data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)
        output = model(data, attention_mask=mask)[1].reshape(-1, 768)
        return output

    def forward(self, material, product, method, method_type, material_type, title, keyword, graph_embed):
        material = self._process_data(material, self.model_material)
        product = self._process_data(product, self.model_product)
        method = self._process_data(method, self.model_method)
        title = self._process_data(title, self.model_title)
        keyword = self._process_data(keyword, self.key_word)
        out = torch.cat([material, product, keyword, method, title, graph_embed], 1)
        # out = self.dropout(out)
        total = self.linear1(out)
        out = self.relu(total)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out

class model_4(nn.Module):
    """
    BERT + Graph
    """
    def __init__(self):
        super(model_4, self).__init__()
        self.model_material = BM.from_pretrained(config.model_name)
        self.model_product = BM.from_pretrained(config.model_name)
        self.model_method = BM.from_pretrained(config.model_name)
        self.model_method_type = BM.from_pretrained(config.model_name)
        self.model_material_type = BM.from_pretrained(config.model_name)
        self.model_title = BM.from_pretrained(config.model_name)
        self.key_word = BM.from_pretrained(config.model_name)
        self.linear_type = nn.Linear(23, 8)
        self.linear1 = nn.Linear(768*5 + 2*5 + 8, 800)
        self.linear2 = nn.Linear(800, 200)
        self.linear3 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def _process_data(self, data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = []
        for sample in data:
            mask.append([1 if i != 0 else 0 for i in sample])
        mask = torch.Tensor(mask).to(device)
        output = model(data, attention_mask=mask)[1].reshape(-1, 768)
        return output

    def forward(self, material, product, method, method_type, material_type, title, keyword, graph_embed):
        material = self._process_data(material, self.model_material)
        product = self._process_data(product, self.model_product)
        method = self._process_data(method, self.model_method)
        title = self._process_data(title, self.model_title)
        keyword = self._process_data(keyword, self.key_word)
        type = self.relu(self.linear_type(torch.cat([method_type, material_type], 1)))
        out = torch.cat([material, product, keyword, method, title, type, graph_embed], 1)
        out = self.dropout(out)
        total = self.linear1(out)
        out = self.relu(total)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        return out

class GCNEncoder(nn.Module):

    def __init__(self, in_channels, hidden_size, out_channels, dropout):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_size, cached=True)
        self.conv2 = GCNConv(hidden_size, out_channels, cached=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.conv2(x, edge_index)
        return out

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class simple_model_tr(nn.Module):
    def __init__(self):
        super(simple_model_tr, self).__init__()
        self.l1 = nn.Linear(36, 200)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x

