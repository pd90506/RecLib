import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from movielens import MovieLens1MDataset

from model import InterpretableModel, NeuralCollaborativeFiltering
from int_module import InterpretModule

import multiprocessing
multiprocessing.set_start_method('spawn', True)

def get_dataset(name, dataset_path, genre_path):
    if name == 'movielens1M':
        return MovieLens1MDataset(dataset_path, genre_path)

    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims

    if name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, int_optimizer, data_loader, criterion, device, log_interval=1000):
    model.train()
    total_loss = 0
    
    # alternative training
    # prediction phase
    for i, (fields, target, genre) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
    # for i, (fields, target, genre) in data_loader:
        fields, target, genre = fields.to(device), target.to(device), genre.to(device)
        y, int_loss = model(fields, genre)
        loss = criterion(y, target.float())
        model.zero_grad()
        all_loss = loss #+ 0.01 * int_loss
        all_loss.backward()
        optimizer.step()
        total_loss += all_loss.item()
        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            # print('    - CEloss:', total_loss / log_interval)
            total_loss = 0
    # alternative training
    # interpretation phase  

    total_loss = 0      
    for i, (fields, target, genre) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
    # for i, (fields, target, genre) in data_loader:
        fields, target, genre = fields.to(device), target.to(device), genre.to(device)
        y, int_loss = model(fields, genre)
        # loss = criterion(y, target.float())
        model.zero_grad()
        all_loss = 0.1 * int_loss
        all_loss.backward()
        int_optimizer.step()
        total_loss += all_loss.item()
        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            # print('    - CEloss:', total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        int_loss = 0
        for fields, target, genre in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target, genre = fields.to(device), target.to(device), genre.to(device)
            y, int_loss = model(fields, genre)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

            importance = model.importance(fields, genre)
        print("The interpretation loss is : {:4f}".format(int_loss.item()))
        print(importance)
    return roc_auc_score(targets, predicts)


def main(dataset_name,
         dataset_path,
         genre_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path, genre_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    # split dataset randomly
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    module = InterpretModule(18, 32).to(device)
    int_model = InterpretableModel(model, module).to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=int_model.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    int_optimizer = torch.optim.Adam(params=int_model.int_module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch_i in range(epoch):
        train(int_model, optimizer, int_optimizer, train_data_loader, criterion, device)
        auc = test(int_model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
    auc = test(int_model, test_data_loader, device)
    print('test auc:', auc)
    torch.save(model, f'{save_dir}/{model_name}.pt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movielens1M')
    parser.add_argument('--dataset_path', default='ml-1m/ratings.dat', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--genre_path', default='ml-1m/genre.dat', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='ncf')
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.genre_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
