## Curriculum Meta-Learning for Next POI Recommendation

The codes for the paper: Curriculum Meta-Learning for Next POI Recommendation (KDD 2021).

This repository is under construction.

### Requirements

```
python 3.6
paddlepaddle >= 2.0.0
numpy >= 1.19.0
x2paddle (please refer to https://github.com/PaddlePaddle/X2Paddle)
```

### Run

To run the codes, one should first put the map search data under the path `./data/dataset/`.
Each city is a `.txt` file, where each line is the search records of one user in that city:

```shell
Each line:
{user_id} \t {os_name} \t {search_record1} \t {search_record2} \t ...
# {os_name}: the operation system of the user's mobile, e.g., "iphone", "andriod", etc.
Each search record:
{time}_{poiid}_{poiloc}_{poiname}_{poitype}_{userloc}_{network}
# {time}: the timestamp when the user searched the POI, e.g., '2021-08-14 10:00:00'
# {poiid}: the POI ID hashcode of the searched POI, e.g., 'wqxc4t8fd147a'
# {poiloc}: the Mercator coordinates of the POI, e.g., '(12967374,4810843)'
# {poiname}: the name of the POI, e.g., 'Happy Cinema'
# {poitype}: the category of the POI, e.g., 'cinema'
# {userloc}: the Mercator coordinates of the user when he/she searched the POI
# {network}: '4G' or 'wifi'
```

Next, set the config files under the path `./data/config/`:

- `base_cities.txt`: The name of all the base cities (for meta-training). Each line for one city name. The city names **should exactly match** the names in `./data/dataset/`. The same below. E.g.,

  ```
  beijing
  shanghai
  shenzhen
  ...
  ```

- `valid_cities.txt`: The name of all the valid cities (for meta-validation).

- `test_cities.txt`: The name of all the test cities (for meta-testing).

- `poi_category.txt`: All the POI categories in the raw dataset. Each line for one POI category. E.g., 

  ```
  cinema
  school
  residential area
  ...
  ```

Finally, you can run the following instructions at `./`:

```shell
sh run_prep.sh
sh run.sh
```

As a fake example, you may run as follows to run our codes on fake dataset:

```sh
python preprocess/fake_data_generator.py
sh run_prep.sh
sh run.sh
# Although the meta-learner cannot learn anything from the fake data..
```

Notice: there are still some warnings during running the codes, but it's just OK to run.



### Cite

Please cite the paper: 

```
@article{Chen2021CurriculumMF,
  title={Curriculum Meta-Learning for Next POI Recommendation},
  author={Yudong Chen and Xin Wang and Miao Fan and Jizhou Huang and Shengwen Yang and Wenwu Zhu},
  journal={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  year={2021}
}
```

