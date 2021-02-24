import numpy as np 
import paddle
from paddle.io import Dataset, DataLoader
import os
import argparse
from cwatcher import *
from tqdm import tqdm

def evaluate(reference_city, target_city, Encoder, Classifier, epoch_num):
    Target_eval = City_Dataset(dataset_type='eval', city_name=target_city)
    Target_eval_loader = DataLoader(dataset=Target_eval)

    # load model param
    root_path = os.path.dirname(os.path.realpath(__file__))
    save_path = root_path + '/../model/ref_' + reference_city + '_epoch' + str(epoch_num) + '/'
    encoder = Encoder()
    classifier = Classifier()
    encoder_state_dict = paddle.load(save_path + 'encoder.pdparams')
    encoder.set_state_dict(encoder_state_dict)
    classifier_state_dict = paddle.load(save_path + 'classifier.pdparams')
    classifier.set_state_dict(classifier_state_dict)
    encoder.eval()
    classifier.eval()

    auc = paddle.metric.Auc()
    for features_T, y_T in tqdm(Target_eval_loader()):
        features_T, y_T = paddle.cast(features_T, dtype='float32'), paddle.cast(y_T, dtype='float32')
        encoded_T = encoder(features_T)
        clf_T = classifier(encoded_T)
        pred_T = np.concatenate((1-clf_T.numpy(), clf_T.numpy()), axis=1)
        y_T = paddle.reshape(y_T, [-1, 1]).numpy()
        auc.update(preds=pred_T, labels=y_T)
    auc_value = auc.accumulate()
    print("AUC:{}".format(auc_value))
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate c-watcher on target city.")
    parser.add_argument("reference_city", type=str)
    parser.add_argument("target_city", type=str)
    parser.add_argument("-e", "--epoch_num", type=int, default="100")
    parser.add_argument("-p", "--other_params", type=str, nargs="*")
    args = parser.parse_args()

    evaluate(reference_city=args.reference_city, \
             target_city=args.target_city, \
             Encoder=eval("Encoder_" + args.reference_city), \
             Classifier=eval("Classifier_" + args.reference_city), \
             epoch_num=args.epoch_num
             )