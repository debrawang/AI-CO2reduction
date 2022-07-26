"""
BERT embedding + Type embedding + Graph embedding
"""
import numpy as np
import torch
import os
import time
from sklearn.linear_model import ARDRegression, LinearRegression, BayesianRidge
from data_loader import *
from final_model import *
from torch import optim
from torch.autograd import Variable
from data_process import *
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer, TrainingArguments
import config as config
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm, linear_model
from datasets import load_dataset
from torch_geometric.nn import GAE, VGAE
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges
from transformers import *
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score#R 2


def _preprocess_sample(sample_str, Z, node_mapping, material_type_mapping, method_type_mapping):
    """
    preprocess each sample with the limitation of maximum length and pad each sample to maximum length
    :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
    :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
    """
    raw_sample = json.loads(sample_str)
    sample = [[] for n in range(len(str_index)+1)]
    for k in raw_sample.keys():
        if k == 'label':
            sample[str_index.index(k)].append(raw_sample[k]/100)
        elif k in ['method_type', 'material_type']:
            mapping = eval(k + '_mapping')
            initial_encode = np.zeros([1, len(mapping)])
            initial_encode[0, mapping[raw_sample[k]]] = 1
            sample[str_index.index(k)] = initial_encode.tolist()[0]
            sample[len(str_index)] += Z[node_mapping[raw_sample[k]]].tolist()
        elif k in ['material', 'product', 'method']:
            sample[str_index.index(k)] = config.tokenizer.encode(raw_sample[k],
                                                                 max_length=str_max_len[str_index.index(k)],
                                                                 pad_to_max_length=True)
            sample[len(str_index)] += Z[node_mapping[raw_sample[k]]].tolist()
        else:
            sample[str_index.index(k)] = config.tokenizer.encode(raw_sample[k],
                                                                 max_length=str_max_len[str_index.index(k)],
                                                                 pad_to_max_length=True)

    return sample


def build_graph_data_from_raw_data():
    raw_data = np.load('./data/raw_data3.npy').tolist()
    value_data = [list(eval(i).values()) for i in raw_data]
    graph_node = []

    # tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    # model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    for one in value_data:
        graph_node += one[:5]
        graph_node += [one[-1]]
    graph_node = np.array(graph_node)
    graph_node_mapping = {index_id: int(i) for i, index_id in enumerate(np.unique(graph_node))}
    # node_x = np.zeros([len(graph_node_mapping), 768])
    node_x = np.zeros([len(graph_node_mapping), 6])
    edge_index = []
    for one_data in raw_data:
        one_data = eval(one_data)
        material = one_data['material']
        node_x[graph_node_mapping[material], 0] = 1
        # node_x[graph_node_mapping[material], :] = model(torch.LongTensor(tokenizer.encode(material)).reshape(1, -1))[1].tolist()[0]
        material_type = one_data['material_type']
        node_x[graph_node_mapping[material_type], 1] = 1
        # node_x[graph_node_mapping[material_type], :] = model(torch.LongTensor(tokenizer.encode(material_type)).reshape(1, -1))[1].tolist()[0]
        product = one_data['product']
        node_x[graph_node_mapping[product], 2] = 1
        # node_x[graph_node_mapping[product], :] = model(torch.LongTensor(tokenizer.encode(product)).reshape(1, -1))[1].tolist()[0]
        method = one_data['method']
        node_x[graph_node_mapping[method], 3] = 1
        # node_x[graph_node_mapping[method], :] = model(torch.LongTensor(tokenizer.encode(method)).reshape(1, -1))[1].tolist()[0]
        method_type = one_data['method_type']
        node_x[graph_node_mapping[method_type], 4] = 1
        # node_x[graph_node_mapping[method_type], :] = model(torch.LongTensor(tokenizer.encode(method_type)).reshape(1, -1))[1].tolist()[0]
        title = one_data['title']
        node_x[graph_node_mapping[title], 5] = 1
        # node_x[graph_node_mapping[title], :] = model(torch.LongTensor(tokenizer.encode(title)).reshape(1, -1))[1].tolist()[0]
        edge_index.append([graph_node_mapping[title], graph_node_mapping[material]])
        edge_index.append([graph_node_mapping[title], graph_node_mapping[product]])
        edge_index.append([graph_node_mapping[title], graph_node_mapping[method]])
        edge_index.append([graph_node_mapping[material], graph_node_mapping[material_type]])
        edge_index.append([graph_node_mapping[method], graph_node_mapping[method_type]])
    rev_edge_index = [[i[1], i[0]] for i in edge_index]
    edge_index = torch.tensor(np.array(edge_index).transpose())
    # rev_edge_index = torch.tensor(np.array(rev_edge_index).transpose())
    data = Data()
    data.num_nodes = len(graph_node_mapping)
    data.edge_index = edge_index
    data.x = torch.ones((data.num_nodes, 15))
    # data.x = torch.tensor(node_x, dtype=torch.float32)
    return data, graph_node_mapping


def train_GAE(model, optimizer, train_pos_edge_index, x):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    # if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test_GAE(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def pre_train_graph():
    data, graph_node_mapping = build_graph_data_from_raw_data()
    data.train_mask = data.val_mask = data.test_mask = None
    data = train_test_split_edges(data)
    out_channels = 2
    num_features = data.num_features
    epochs = 100
    hidden_size = 30
    dropout = 0.2
    # model
    # model = GAE(GCNEncoder(num_features, hidden_size, out_channels, dropout))
    model = VGAE(VariationalGCNEncoder(num_features, out_channels))
    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    # inizialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, epochs + 1):
        loss = train_GAE(model, optimizer, train_pos_edge_index, x)
        auc, ap = test_GAE(model, x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, LOSS:{:.4f}'.format(epoch, auc, ap, loss))
    Z = model.encode(x, train_pos_edge_index)
    return Z, graph_node_mapping


def validate(device, val_loader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_emb, target,one_hot_embedding) in enumerate(
                val_loader):
            material = Variable(material).to(device)
            product = Variable(product).to(device)
            method = Variable(method).to(device)
            method_type = Variable(method_type).to(device)
            material_type = Variable(material_type).to(device)
            title = Variable(title).to(device)
            key_word = Variable(key_word).to(device)
            target = Variable(target.view(-1, 1)).to(device)
            graph_emb = Variable(graph_emb).to(device)
            # mask = []
            # for sample in data:
            #     mask.append([1 if i != 0 else 0 for i in sample])
            # mask = torch.Tensor(mask).to(device)
            output = model(material, product, method, method_type, material_type, title, key_word, graph_emb)
            loss = criterion(output, target)
            val_loss.append(loss.item())
        return np.mean(val_loss)


def build_data():
    graph_embedding_Z, node_mapping = pre_train_graph()
    corpus_file = load_data_from_npy(config.raw_data_path + '/raw_data3.npy')
    all_sample = generate_onehot(corpus_file)
    one_hot_embedding, input_labels = [i[0] for i in all_sample], [i[1] for i in all_sample]
    material_type = np.unique(np.array([eval(i)['material_type'] for i in corpus_file]))
    material_type_mapping = {index_id: int(i) + 0 for i, index_id in enumerate(np.unique(material_type))}
    method_type = np.unique(np.array([eval(i)['method_type'] for i in corpus_file]))
    method_type_mapping = {index_id: int(i) + 0 for i, index_id in enumerate(np.unique(method_type))}
    token_file = [_preprocess_sample(i, graph_embedding_Z, node_mapping, material_type_mapping, method_type_mapping) for
                  i in corpus_file]
    material = [i[0] for i in token_file]
    product = [i[1] for i in token_file]
    method = [i[2] for i in token_file]
    method_type = [i[3] for i in token_file]
    material_type = [i[4] for i in token_file]
    title = [i[5] for i in token_file]
    keyword = [i[6] for i in token_file]
    label = [i[7] for i in token_file]
    graph_embed = [i[8] for i in token_file]
    all_dataset = TensorDataset(torch.LongTensor(material),
                                torch.LongTensor(product),
                                torch.LongTensor(method),
                                torch.FloatTensor(method_type),
                                torch.FloatTensor(material_type),
                                torch.LongTensor(title),
                                torch.LongTensor(keyword),
                                torch.FloatTensor(graph_embed),
                                torch.FloatTensor(label),
                                torch.FloatTensor(one_hot_embedding))
    train_len = int(len(all_dataset) * 0.7)
    test_len = int(len(all_dataset) * 0.15)
    valid_len = len(all_dataset) - train_len - test_len
    train_dataset, valid_dataset, test_dataset = random_split(all_dataset, [train_len, valid_len, test_len])

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=3,
                              shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=3,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=True)
    return train_loader, test_loader,valid_loader,material_type_mapping,method_type_mapping


def _preprocess_sample_tradition(sample_str, str_type):
    """
    preprocess each sample with the limitation of maximum length and pad each sample to maximum length
    :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
    :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
    """
    data_type = np.array([json.loads(i)[str_type] for i in sample_str])
    data_type_encoder = LabelEncoder()
    data_type_encoded = data_type_encoder.fit_transform(data_type)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = data_type_encoded.reshape(len(data_type_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


def generate_onehot(corpus_file):
    sample = []
    method_type = _preprocess_sample_tradition(corpus_file, 'method_type')
    material_type = _preprocess_sample_tradition(corpus_file, 'material_type')
    product = _preprocess_sample_tradition(corpus_file, 'product')
    for data_id, data in enumerate(corpus_file):
        data = json.loads(data)
        label = 0
        data_one = []
        for k in data.keys():
            if k == 'label':
                label = data[k] / 100
            elif k in ['method_type', 'material_type', 'product']:
                data_one_ = eval(k + '[data_id].tolist()')
                data_one += data_one_
            else:
                pass
        sample.append([data_one, [label]])
    return sample


def train(model_type, train_loader, test_loader,valid_loader,path):
    """

    :param model_type: 1: BERT 2: BERT + Type Embedding 3:BERT + Graph Embedding 4:ALL
    :return:
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simple_model_ = eval('model_' + str(model_type) + '()')
    simple_model_.to(device)
    simple_model_.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(simple_model_.parameters(), lr=1e-5)
    best_valid_loss = 100
    early_stop = 0
    Loss = []
    for i in range(config.epoch):
        for batch_idx, (
        material, product, method, method_type, material_type, title, key_word, graph_embed, target, one_hot_embedding) in enumerate(
                train_loader):
            material = Variable(material).to(device)
            product = Variable(product).to(device)
            method = Variable(method).to(device)
            method_type = Variable(method_type).to(device)
            material_type = Variable(material_type).to(device)
            title = Variable(title).to(device)
            key_word = Variable(key_word).to(device)
            graph_embed = Variable(graph_embed).to(device)
            target = Variable(target.view(-1, 1)).to(device)
            # mask = []
            # for sample in data:
            #     mask.append([1 if i != 0 else 0 for i in sample])
            # mask = torch.Tensor(mask).to(device)
            output = simple_model_(material, product, method, method_type, material_type, title, key_word, graph_embed)
            pred = output
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ((batch_idx + 1) % config.accumulation_steps) == 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    i + 1, batch_idx, len(train_loader), 100. *
                    batch_idx / len(train_loader), loss.item()
                ))
                running_loss = loss.item()
                Loss.append(running_loss)
            if batch_idx == len(train_loader) - 1:
                print('labels:', target)
                print('pred:', pred)
            val_loss = validate(device, valid_loader, simple_model_, criterion)
            print('val_loss:' + str(val_loss))
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                torch.save(simple_model_.state_dict(), path + str(model_type) + 'model.pt')
                early_stop = 0
            else:
                early_stop += 1
            if early_stop >= 10:
                break
    del simple_model_
    torch.cuda.empty_cache()
    Loss0 = torch.tensor(Loss)
    torch.save(Loss0, path + str(model_type) + 'epoch_{}'.format(epoch))
    model_best = eval('model_' + str(model_type) + '()')
    model_best.load_state_dict(state_dict=torch.load(path + str(model_type) + 'model.pt'))
    model_best.to(device)
    total_loss = 0
    correct = 0
    total = 0
    test_result = []
    with torch.no_grad():
        for batch_idx, (
                material, product, method, method_type, material_type, title, key_word, graph_embed,
                target, one_hot_embedding) in enumerate(test_loader):
            material = Variable(material).to(device)
            product = Variable(product).to(device)
            method = Variable(method).to(device)
            method_type = Variable(method_type).to(device)
            material_type = Variable(material_type).to(device)
            title = Variable(title).to(device)
            key_word = Variable(key_word).to(device)
            graph_embed = Variable(graph_embed).to(device)
            target = Variable(target.view(-1, 1)).to(device)
            # mask = []
            # for sample in data:
            #     mask.append([1 if i != 0 else 0 for i in sample])
            # mask = torch.Tensor(mask).to(device)
            output = model_best(material, product, method, method_type, material_type, title, key_word, graph_embed)
            test_result.append([output.tolist()[0], target.tolist()[0]])
            print(output, target)
            total_loss += criterion(output, target)
            if abs(output - target) < 0.1:
                correct += 1
            total += 1
    y_test = [i[1] for i in test_result]
    y_predict = [i[0] for i in test_result]
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test,y_predict)
    print(total_loss / total)
    print(correct / total)
    del model_best
    torch.cuda.empty_cache()
    return Loss0, test_result, [mse, mae, r2]


def compare_train(model_type, train_loader, test_loader):
    # train_loader, test_loader = data_loaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    test_result = []
    if model_type == 'MLP':
        simple_model_ = simple_model_tr()
        simple_model_.to(device)
        simple_model_.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(simple_model_.parameters(), lr=1e-5)
        for i in range(config.epoch):
            for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_embed,
                target, one_hot_embedding) in enumerate(train_loader):
                data, target = Variable(one_hot_embedding).to(device), Variable(target.view(-1, 1)).to(device)
                output = simple_model_(data)
                pred = output
                loss = criterion(output, target)

                loss = loss / config.accumulation_steps
                loss.backward()

                if ((batch_idx + 1) % config.accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                if ((batch_idx + 1) % config.accumulation_steps) == 1:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                        i + 1, batch_idx, len(train_loader), 100. *
                        batch_idx / len(train_loader), loss.item()
                    ))
                if batch_idx == len(train_loader) - 1:
                    print('labels:', target)
                    print('pred:', pred)
        simple_model_.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_embed,
                target, one_hot_embedding) in enumerate(test_loader):
                data = one_hot_embedding.to(device)
                target = target.to(device)
                mask = []
                for sample in data:
                    mask.append([1 if i != 0 else 0 for i in sample])
                output = simple_model_(data)
                test_result.append([output.tolist()[0], target.tolist()[0]])
                print(output, target)
                total_loss += criterion(output, target)
                if abs(output - target) < 0.1:
                    correct += 1
                total += 1

            print(total_loss / total)
            print(correct / total)
    if model_type == 'SVR':
        total_loss = 0
        correct = 0
        total = 0
        regr = svm.SVR()
        for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_embed,
                        target, one_hot_embedding) in enumerate(train_loader):
            regr.fit(one_hot_embedding.tolist(), target.reshape(1, -1).tolist()[0])
        for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_embed,
                        target, one_hot_embedding) in enumerate(test_loader):
            output = torch.tensor(regr.predict(one_hot_embedding.tolist()))
            test_result.append([output.tolist()[0], target.tolist()[0]])
            print(output, target)
            total_loss += criterion(output, target)
            if abs(output - target) < 0.1:
                correct += 1
            total += 1
        print(total_loss / total)
        print(correct / total)

    if model_type == 'LinearRegression':
        total_loss = 0
        correct = 0
        total = 0
        regr = LinearRegression()
        for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_embed,
                        target, one_hot_embedding) in enumerate(train_loader):
            regr.fit(one_hot_embedding.tolist(), target.reshape(1, -1).tolist()[0])
        for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_embed,
                        target, one_hot_embedding) in enumerate(test_loader):
            output = torch.tensor(regr.predict(one_hot_embedding.tolist()))
            print(output, target)
            test_result.append([output.tolist()[0], target.tolist()[0]])
            total_loss += criterion(output, target)
            if abs(output - target) < 0.1:
                correct += 1
            total += 1
        print(total_loss / total)
        print(correct / total)
    if model_type == 'Bayesian Ridge Regression':
        total_loss = 0
        correct = 0
        total = 0
        regr = linear_model.BayesianRidge()
        for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_embed,
                        target, one_hot_embedding) in enumerate(train_loader):
            regr.fit(one_hot_embedding.tolist(), target.reshape(1, -1).tolist()[0])
        for batch_idx, (material, product, method, method_type, material_type, title, key_word, graph_embed,
                        target, one_hot_embedding) in enumerate(test_loader):
            output = torch.tensor(regr.predict(one_hot_embedding.tolist()))
            print(output, target)
            test_result.append([output.tolist()[0], target.tolist()[0]])
            total_loss += criterion(output, target)
            if abs(output - target) < 0.1:
                correct += 1
            total += 1
        print(total_loss / total)
        print(correct / total)
    y_test = [i[1] for i in test_result]
    y_predict = [i[0] for i in test_result]
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)
    return test_result, [mse, mae, r2]









if __name__ == '__main__':
    for i in range(100):
        t = str(time.time()).replace('.', '_')
        print(str(t))
        os.mkdir(str(t))
        path = str(t) + '/'
        train_loader, test_loader,valid_loader,material_type_mapping,method_type_mapping = build_data()
        print(material_type_mapping)
        print(method_type_mapping)
        # torch.save(train_loader, path + str(t) + 'train_loader.pt')
        # torch.save(test_loader, path + str(t) + 'test_loader.pt')
        # train_loader = torch.load(path + str(t) + 'train_loader.pt')
        # test_loader = torch.load(path + str(t) + 'test_loader.pt')
        # test_svr, loss_svr = compare_train('SVR', train_loader, test_loader)
        # test_lr, loss_lr = compare_train('LinearRegression', train_loader, test_loader)
        # test_brr, loss_brr = compare_train('Bayesian Ridge Regression', train_loader, test_loader)
        # test_mlp, loss_mlp = compare_train('MLP', train_loader, test_loader)
        print("just BERT")
        train_loss_1, test_result_1, test_loss_1 = train(1, train_loader, test_loader, valid_loader,path)
        print("BERT + Type")
        train_loss_2, test_result_2, test_loss_2 = train(2, train_loader, test_loader, valid_loader,path)
        # print("BERT + Graph")
        # train_loss_3, test_result_3, test_loss_3 = train(3, train_loader, test_loader, valid_loader,path)
        # print("all")
        # train_loss_4, test_result_4, test_loss_4 = train(4, train_loader, test_loader, valid_loader,path)
        # print('SVR' + str(loss_svr))
        # print('LinearRegression' + str(loss_lr))
        # print('Bayesian Ridge Regression' + str(loss_brr))
        # print('MLP' + str(loss_mlp))
        # print("just BERT:" + str(test_loss_1))
        # print("BERT + Type:" + str(test_loss_2))
        # print("BERT + Graph:" + str(test_loss_3))
        # print("ALL:" + str(test_loss_4))
        with open(path + 'result.txt', 'w+') as f:
            f.write('just BERT\n')
            f.write('train_loss\n')
            f.write(str(train_loss_1) + '\n')
            f.write('test_result\n')
            f.write(str(test_result_1) + '\n')
            f.write('test_loss\n')
            f.write(str(test_loss_1) + '\n')
            f.write('BERT + Type:\n')
            f.write('train_loss\n')
            f.write(str(train_loss_2) + '\n')
            f.write('test_result\n')
            f.write(str(test_result_2) + '\n')
            f.write('test_loss\n')
            f.write(str(test_loss_2) + '\n')
            # f.write('BERT + Graph\n')
            # f.write('train_loss\n')
            # f.write(str(train_loss_3) + '\n')
            # f.write('test_result\n')
            # f.write(str(test_result_3) + '\n')
            # f.write('test_loss\n')
            # f.write(str(test_loss_3) + '\n')
            # f.write('ALL\n')
            # f.write('train_loss\n')
            # f.write(str(train_loss_4) + '\n')
            # f.write('test_result\n')
            # f.write(str(test_result_4) + '\n')
            # f.write('test_loss\n')
            # f.write(str(test_loss_4) + '\n')
            # f.write('SVR\n')
            # f.write('test_result\n')
            # f.write(str(test_svr) + '\n')
            # f.write('test_loss\n')
            # f.write(str(loss_svr) + '\n')
            # f.write('LinearRegression\n')
            # f.write('test_result\n')
            # f.write(str(test_lr) + '\n')
            # f.write('test_loss\n')
            # f.write(str(loss_lr) + '\n')
            # f.write('Bayesian Ridge Regression\n')
            # f.write('test_reslut\n')
            # f.write(str(test_brr) + '\n')
            # f.write('test_loss\n')
            # f.write(str(loss_brr) + '\n')
            # f.write('MLP\n')
            # f.write('test_result\n')
            # f.write(str(test_mlp) + '\n')
            # f.write('test_loss\n')
            # f.write(str(loss_mlp) + '\n')
            f.write(str(material_type_mapping) + '\n')
            f.write(str(method_type_mapping) + '\n')


