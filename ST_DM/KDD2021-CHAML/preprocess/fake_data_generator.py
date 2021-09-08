import paddle
import os
import random
root_path = './data/'
dataset_path = root_path + 'dataset/'
config_path = root_path + 'config/'
city_nums = [8, 2, 4]
title = ['mtrain', 'mvalid', 'mtest']
title2 = ['base_cities', 'valid_cities', 'target_cities']
city_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
user_num = 1200
poi_num = 300
poi_type_num = 229
record_num = 10
time_str = '2021-08-14 10:00:00'
user_loc = '(777,888)'
poi_loc = '(778,889)'
poi_name = 'fake'


def generate_city(file_name):
    with open(file_name, 'w') as f:
        for u in range(user_num):
            line = str(u) + '\tx'
            for r in range(record_num):
                poi_id = 'poiid' + str(random.randint(1, poi_num))
                poi_type = 'poitype' + str(random.randint(1, poi_type_num))
                line += '\t' + '_'.join([time_str, poi_id, poi_loc,
                    poi_name, poi_type, user_loc, 'x'])
            f.write(line + '\n')


def generate_poitype_config():
    with open(config_path + 'poi_category.txt', 'w') as f:
        for t in range(1, poi_type_num + 1):
            f.write('poitype_' + str(t) + '\n')


def generate_city_names_config(names, role):
    with open(config_path + title2[role] + '.txt', 'w') as f:
        for name in names:
            f.write(name + '\n')


if __name__ == '__main__':

    def go():
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        if not os.path.exists(config_path):
            os.mkdir(config_path)
        cnt = 0
        for i in range(len(city_nums)):
            names = []
            for j in range(city_nums[i]):
                city_name = title[i] + '-' + city_names[cnt]
                file_name = dataset_path + city_name + '.txt'
                generate_city(file_name)
                names.append(city_name)
                cnt += 1
            generate_city_names_config(names, i)
        generate_poitype_config()
    go()
