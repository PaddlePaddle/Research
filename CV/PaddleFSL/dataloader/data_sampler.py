import numpy as np
import paddle
import paddle.fluid as fluid

class BatchSampler():
    def __init__(self, data, args):
        self.data = data
        self.dataset = args.dataset
        if args.test_mode:
            self.n_episodes = args.test_episodes
        else:
            self.n_episodes = args.episodes
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_query = args.n_query

        label = np.array(data.label)
        self.label_unique = np.unique(label)
        self.c_ind = []
        for i in range(max(self.label_unique)+1):
            ind = np.argwhere(label == i).reshape(-1)
            self.c_ind.append(ind)
        self.n_classes = len(self.label_unique)
    
    def __iter__(self):
        for _ in range(self.n_episodes):
            batch = []
            selected_classes_ind = np.random.permutation(self.n_classes)[:self.n_way]
            selected_classes = self.label_unique[selected_classes_ind]
            for c in selected_classes:
                selected_img_ind = np.random.permutation(len(self.c_ind[c]))[:(self.k_shot+self.n_query)]
                batch.append(self.c_ind[c][selected_img_ind])
            selected_index = np.stack(batch).reshape(-1)
            batch_data,_ = self.data[selected_index]
            batch_data = batch_data.astype(np.float32)
            batch_label = np.tile(np.arange(self.n_way).reshape(self.n_way,1), (1, self.k_shot+self.n_query)).flatten()
            yield batch_data, batch_label