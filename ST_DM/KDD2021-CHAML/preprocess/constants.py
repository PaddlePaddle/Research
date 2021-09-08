MAX_HIST = 10
MIN_HIST = 5
SPT_SIZE = 2
MAX_USER_TOTAL = 2000
TRAIN_USER_NUM = 2000
TEST_USER_NUM = 500
FULL_NEG_SAMPLING = False
root_path = './data/'
dataset_path = root_path + 'dataset/'
split_path = dataset_path + 'split/'
save_path = dataset_path + 'final/'
config_path = root_path + 'config/'


def get_cities(which='base'):
    cities_file = config_path + which + '_cities.txt'
    cities = []
    with open(cities_file, 'r') as f:
        for line in f:
            city = line.strip()
            cities.append(city)
    return cities


def city_user_id2idx_path(city_name):
    return root_path + 'pkls/' + city_name + '/userid_to_id.pkl'
