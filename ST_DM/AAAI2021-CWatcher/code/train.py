import numpy as np 
import paddle
from paddle.io import Dataset, DataLoader
import os
import argparse
from cwatcher import *
from tqdm import tqdm

def train(reference_city, Encoder, Decoder, Discriminator, Classifier, epoch_num, lr):
    np.random.seed(2)
    paddle.seed(2)
    Wuhan_train = City_Dataset(dataset_type='train', city_name='wuhan.labeled')
    Target_train = City_Dataset(dataset_type='train', city_name=reference_city)
    Wuhan_train_loader = DataLoader(dataset=Wuhan_train, batch_size=64, shuffle=True)
    Target_train_loader = DataLoader(dataset=Target_train, batch_size=64, shuffle=True)

    # instance of each part
    encoder = Encoder()
    decoderE = Decoder()  # decoder for epicenter city Wuhan
    decoderT = Decoder()  # decoder for target city
    discriminator = Discriminator()
    classifier = Classifier()

    # init 2 types of loss fuction 
    loss_bn = paddle.nn.BCELoss()  
    loss_mse = paddle.nn.MSELoss()

    # set optimizer for each part of the frameworwk
    optimizer_discrimitator = paddle.optimizer.Adam(parameters=discriminator.parameters(), learning_rate = lr)
    optimizer_encoder = paddle.optimizer.Adam(parameters=encoder.parameters(), learning_rate = lr)
    optimizer_decoderE = paddle.optimizer.Adam(parameters=decoderE.parameters(), learning_rate = lr)
    optimizer_decoderT = paddle.optimizer.Adam(parameters=decoderT.parameters(), learning_rate = lr)
    optimizer_classifier = paddle.optimizer.Adam(parameters=classifier.parameters(), learning_rate = lr)

    for epoch in tqdm(range(epoch_num)):

        encoder.train()
        decoderE.train()
        decoderT.train()
        classifier.train()
        discriminator.train()

        for features_T, y_T in Target_train_loader():
            try:
                features_E, y_E = epicenter_iter.next()
            except:
                epicenter_iter = iter(Wuhan_train_loader())
                features_E, y_E = epicenter_iter.next()
            features_T, y_T = paddle.cast(features_T, dtype='float32'), paddle.cast(y_T, dtype='float32')
            features_E, y_E = paddle.cast(features_E, dtype='float32'), paddle.cast(y_E, dtype='float32')

            is_epicenter = paddle.full([features_E.shape[0]], 1.0)  
            not_epicenter = paddle.full([features_T.shape[0]], 0.0)
            fake_epicenter = paddle.full([features_T.shape[0]], 1.0)

            encoded_E = encoder(features_E)  # encoded representation of neighborhoods from epicenter
            encoded_T = encoder(features_T)  # encoded representation of neighborhoods from target city
            
            loss_cheat_E = loss_bn(paddle.reshape(discriminator(encoded_E), [-1]), is_epicenter)
            loss_cheat_T = loss_bn(paddle.reshape(discriminator(encoded_T), [-1]), fake_epicenter)
            loss_cheat = (loss_cheat_E + loss_cheat_T) * 0.5
        
            decoded_E = decoderE(encoded_E)
            decoded_T = decoderT(encoded_T)
            loss_recons_E = loss_mse(decoded_E, features_E)
            loss_recons_T = loss_mse(decoded_T, features_T)
            loss_recons = (loss_recons_E + loss_recons_T) * 0.5
            
            clf_E = paddle.reshape(classifier(encoded_E), [-1])
            loss_clf = loss_bn(clf_E, y_E)
            
            # optimize encoder
            optimizer_encoder.clear_grad()
            loss_encoder = 0.7 *loss_cheat + 0.1 *loss_recons + 0.2 *loss_clf
            loss_encoder.backward()
            optimizer_encoder.step()

            # train decoders & clf
            encoded_E = encoded_E.detach()
            encoded_T = encoded_T.detach()
            decoded_E = decoderE(encoded_E)
            decoded_T = decoderT(encoded_T)
            clf_E = paddle.reshape(classifier(encoded_E), [-1])
            
            optimizer_decoderE.clear_grad()
            loss_recons_E = loss_mse(decoded_E, features_E)
            loss_recons_E.backward()
            optimizer_decoderE.step()
            
            optimizer_decoderT.clear_grad()
            loss_recons_T = loss_mse(decoded_T, features_T)
            loss_recons_T.backward()
            optimizer_decoderT.step()

            optimizer_classifier.clear_grad()
            loss_clf = loss_bn(clf_E, y_E)
            loss_clf.backward()
            optimizer_classifier.step()	   

            # train discriminator
            disc_E = paddle.reshape(discriminator(encoded_E), [-1])
            disc_T = paddle.reshape(discriminator(encoded_T), [-1])
            loss_diff_E = loss_bn(disc_E, is_epicenter)
            loss_diff_T = loss_bn(disc_T, not_epicenter)
            loss_diff = (loss_diff_E + loss_diff_T) * 0.5
            
            optimizer_discrimitator.clear_grad()
            loss_diff.backward()
            optimizer_discrimitator.step()

    # save model
    root_path = os.path.dirname(os.path.realpath(__file__))
    save_path = root_path + '/../model/ref_' + reference_city + '_epoch' + str(epoch_num) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    paddle.save(encoder.state_dict(), save_path + "encoder.pdparams")
    paddle.save(decoderE.state_dict(), save_path + "decoderE.pdparams")
    paddle.save(decoderT.state_dict(), save_path + "decoderT.pdparams")
    paddle.save(discriminator.state_dict(), save_path + "discriminator.pdparams")
    paddle.save(classifier.state_dict(), save_path + "classifier.pdparams")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train c-watcher on epicenter and reference city.")
    parser.add_argument("reference_city", type=str)
    parser.add_argument("-e", "--epoch_num", type=int, default="100")
    parser.add_argument("-lr", "--learning_rate", type=float, default="0.0001")
    parser.add_argument("-p", "--other_params", type=str, nargs="*")
    args = parser.parse_args()
    
    train(reference_city=args.reference_city, \
          Encoder=eval("Encoder_" + args.reference_city), \
          Decoder=eval("Decoder_" + args.reference_city), \
          Discriminator=eval("Discriminator_" + args.reference_city), \
          Classifier=eval("Classifier_" + args.reference_city), \
          epoch_num=args.epoch_num, \
          lr=args.learning_rate
          )