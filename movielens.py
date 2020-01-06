import numpy as np
import pandas as pd
import torch.utils.data


class MovieLens1MDataset(torch.utils.data.Dataset):
    """
    MovieLens 1M Dataset

    :param dataset_path: MovieLens dataset path

    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, genre_path, sep='::'):
        data = pd.read_csv(dataset_path, sep=sep, engine='python', header=None).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        # self.targets = data[:, 2].astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

        # get genre info for movielens 1m dataset
        genre = Movielens1MGenre(genre_path)
        self.genre_dict = genre.getGenre()
        
        

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        movie_id = self.items[index, 1] # get the movie id to obtain genre info
        return self.items[index], self.targets[index] , self.genre_dict[movie_id]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


# class MovieLens1MDataset(MovieLens20MDataset):
#     """
#     MovieLens 1M Dataset

#     :param dataset_path: MovieLens dataset path

#     Reference:
#         https://grouplens.org/datasets/movielens
#     """

#     def __init__(self, dataset_path):
#         super().__init__(dataset_path, '::')


class Movielens1MGenre():
    """
    :param movie_path: the path of movies.dat
    """

    def __init__(self, genre_path, sep=','):
        genre_data = pd.read_csv(genre_path, sep=sep, engine='python', header=None).to_numpy()
        self.genreDict = self._getGenreDict(genre_data)

    def getGenre(self):
        return self.genreDict

    def _getGenreDict(self, genre_data):
        ids = genre_data[:, 0].astype(np.int)
        genre_arrays = genre_data[:, 1:].astype(np.float32)
        genre_dict = {ids[i]: genre_arrays[i, :] for i in range(len(ids))}
        return genre_dict




if __name__ == "__main__":
    data_path = "ml-1m/ratings.dat"
    genre_path = "ml-1m/genre.dat"
    a = MovieLens1MDataset(data_path, genre_path)
    # path = "ml-1m/genre.dat"
    # a = Movielens1MGenre(path)

    print()