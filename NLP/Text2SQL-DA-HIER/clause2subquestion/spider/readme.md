## Question Generation
### train

We trained our model with OpenNMT: https://github.com/OpenNMT/OpenNMT-py

version: v1.1.1 (https://github.com/OpenNMT/OpenNMT-py/tree/1.1.1)

### generate data

    # get sub-sql and sub-question
    cd clause2subquestion
    python data.py # training data


    # get source data for sql2question
    cd spider
    python aug2src.py # augment data

### model train and prediction

Details refer to https://github.com/OpenNMT/OpenNMT-py. Our code can be found in director clause2subquestion/ .

Our generated data for Spider can be found at https://aistudio.baidu.com/aistudio/datasetdetail/123584 .