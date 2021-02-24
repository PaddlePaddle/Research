import numpy as np 
import paddle
from paddle.io import Dataset, DataLoader
import os


# read data
class City_Dataset(Dataset):
    def __init__(self, dataset_type, city_name):
        super(City_Dataset, self).__init__()
        self.type = dataset_type
        self.city_name = city_name
        root_path = os.path.dirname(os.path.realpath(__file__))
        self.data_root = root_path + '/../data/' + self.type + '_' + self.city_name
        self.data_list = []
        
        with open(self.data_root, 'r') as f:
            for line in f:
                line = eval(line.strip('\n'))
                self.data_list.append(line)

    def __getitem__(self, index):
        id, features, label = self.data_list[index]
        features = np.array(features)
        label = int(label) 
        return features, label

    def __len__(self):
        return len(self.data_list)


# reference city is Shenzhen
class Encoder_shenzhen(paddle.nn.Layer):
    def __init__(self):
        super(Encoder_shenzhen, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 16)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class Decoder_shenzhen(paddle.nn.Layer):
    def __init__(self):
        super(Decoder_shenzhen, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class Discriminator_shenzhen(paddle.nn.Layer):
    def __init__(self):
        super(Discriminator_shenzhen, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class Classifier_shenzhen(paddle.nn.Layer):
    def __init__(self):
        super(Classifier_shenzhen, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y


# reference city is Changsha
class Encoder_changsha(paddle.nn.Layer):
    def __init__(self):
        super(Encoder_changsha, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 1024)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class Decoder_changsha(paddle.nn.Layer):
    def __init__(self):
        super(Decoder_changsha, self).__init__()
        self.linear_1 = paddle.nn.Linear(1024, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class Discriminator_changsha(paddle.nn.Layer):
    def __init__(self):
        super(Discriminator_changsha, self).__init__()
        self.linear_1 = paddle.nn.Linear(1024, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class Classifier_changsha(paddle.nn.Layer):
    def __init__(self):
        super(Classifier_changsha, self).__init__()
        self.linear_1 = paddle.nn.Linear(1024, 32)
        self.linear_2 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y


# reference city is Shanghai
class Encoder_shanghai(paddle.nn.Layer):
    def __init__(self):
        super(Encoder_shanghai, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 16)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class Decoder_shanghai(paddle.nn.Layer):
    def __init__(self):
        super(Decoder_shanghai, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class Discriminator_shanghai(paddle.nn.Layer):
    def __init__(self):
        super(Discriminator_shanghai, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class Classifier_shanghai(paddle.nn.Layer):
    def __init__(self):
        super(Classifier_shanghai, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 32)
        self.linear_2 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y


# reference city is Zhengzhou
class Encoder_zhengzhou(paddle.nn.Layer):
    def __init__(self):
        super(Encoder_zhengzhou, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 128)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class Decoder_zhengzhou(paddle.nn.Layer):
    def __init__(self):
        super(Decoder_zhengzhou, self).__init__()
        self.linear_1 = paddle.nn.Linear(128, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class Discriminator_zhengzhou(paddle.nn.Layer):
    def __init__(self):
        super(Discriminator_zhengzhou, self).__init__()
        self.linear_1 = paddle.nn.Linear(128, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class Classifier_zhengzhou(paddle.nn.Layer):
    def __init__(self):
        super(Classifier_zhengzhou, self).__init__()
        self.linear_1 = paddle.nn.Linear(128, 32)
        self.linear_2 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y


# reference city is Chengdu
class Encoder_chengdu(paddle.nn.Layer):
    def __init__(self):
        super(Encoder_chengdu, self).__init__()
        self.linear_1 = paddle.nn.Linear(236, 16)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.relu(y)
        return y

class Decoder_chengdu(paddle.nn.Layer):
    def __init__(self):
        super(Decoder_chengdu, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 236)
        self.tanh = paddle.nn.Tanh()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.tanh(y)
        return y

class Discriminator_chengdu(paddle.nn.Layer):
    def __init__(self):
        super(Discriminator_chengdu, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 16)
        self.linear_2 = paddle.nn.Linear(16, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y

class Classifier_chengdu(paddle.nn.Layer):
    def __init__(self):
        super(Classifier_chengdu, self).__init__()
        self.linear_1 = paddle.nn.Linear(16, 32)
        self.linear_2 = paddle.nn.Linear(32, 1)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.5)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, inputs):
        y = self.dropout(inputs)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        return y





