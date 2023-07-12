
import numpy as np
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Flatten, concatenate
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from dataset import Dataset
from eval import evaluate_model
import time 
import multiprocessing as mp
import sys
import math
import argparse
import torch
import os
import wandb
from pathlib import Path
from config import MODEL_PARAMETERS, LEARNING_RATE, BATCH_SIZE, TOPK, NUM_NEGATIVES
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(
        prog='RecSys',
        description="Recommendation system based on matrix factorization")
    params = MODEL_PARAMETERS['MLP']
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument('-lr', '--learning_rate', default=LEARNING_RATE, metavar='learning-rate',
                        help='learning rate value for model training')
    parser.add_argument('--reg_layers', nargs='?', default=params['reg_layers'],
                        help="Regularization for each layer")
    parser.add_argument('--layers', nargs='?', default=params["layers"],
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, metavar='batch-size',
                        help='batch size for local model at the user-level')
    parser.add_argument('-k', '--top_k', default=TOPK, metavar='top-k',
                        help='batch size for local model at the user-level')
    parser.add_argument('--num_neg', type=int, default=NUM_NEGATIVES,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('-e', '--epochs', default=400, metavar='epochs', type=int,
                        help='number of training epochs, default 500')
    parser.add_argument('-d', '--dataset', default='movielens', metavar='dataset',
                        choices=['movielens', 'amazon' ,'foursquare'],
                        help='which dataset to use, default "movielens"')
    parser.add_argument('-p', '--path', nargs='?', default='./../../dataset',
                        help='Input data path.')
    parser.add_argument('-o', '--out', nargs='?', default='./pretrained',
                        help='Output data path.')
    parser.add_argument('-v', '--validation_steps', default=10, type=int)
    parser.add_argument('-l', '--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('-n', '--name', default='NeuCF', metavar='name',
                        help='name of the model')
    args, leftovers = parser.parse_known_args()
    if args.dataset == 'movielens': 
        parser.add_argument(
        "--type",
        type=str,
        default="ml-100k",
        choices=["ml-1m", "ml-25m", "ml-100k"],
        help="decide which type of movielens dataset: ml-1m, ml-25m or ml-100k",)
    if args.dataset == 'amazon': 
        parser.add_argument(
        "--type",
        type=str,
        default="grocery",
        choices=["grocery", "fashion", "ml-100k"],
        help="decide which type of Amazon products",)
    if args.dataset == 'foursquare': 
        parser.add_argument(
        "--type",
        type=str,
        default="nyc",
        choices=["nyc", "tky"],
        help="decide which type of foursquare dataset",)
 
    
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializers.normal(shape, scale=0.01, name=name)


def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = 'user_embedding',
                                   embeddings_regularizer= l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'item_embedding',
                                   embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten() (MLP_Embedding_User(user_input))
    item_latent = Flatten() (MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = concatenate([user_latent, item_latent])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    num_items = train.shape[1]

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':

    args = parse_args()
    raw_path = Path(args.path) / args.dataset / args.type
    out_path = Path(args.out) / args.dataset / args.type
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.validation_steps
    
    topK = args.top_k
    evaluation_threads = 1 #mp.cpu_count()
    #print("GMF arguments: %s" %(args))
    model_out_file = 'pretrained/%s/%s/%s_MLP_%s.h5' %(args.dataset, args.type, args.type, args.layers)
    
    # Loading data
    t1 = time.time()
    dataset = Dataset(raw_path)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time.time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad( learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop( learning_rate=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam( learning_rate=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD( learning_rate=learning_rate), loss='binary_crossentropy')
    #print(model.summary())
    # Init performance
    t = time.time()
    (hits, ndcgs, apks, arks) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg, mapk, mark = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(apks).mean(), np.array(arks).mean()
    #mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
    #p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time.time()-t1))
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    user_input, item_input, labels = get_train_instances(train, num_negatives)
    wandb.init(
            project=f"{args.dataset}-{args.type}", config=args, entity=args.wandb_entity, name=f"{args.name}"
        )
    
    for epoch in range(epochs):
        print('----------- Itration {} -----------'.format(epoch+1))
        t1 = time.time()
        # Generate training instances
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True, use_multiprocessing = True)
        print('Iteration %d [%.1f s]: loss = %.4f  ' 
                  % (epoch+1, time.time()-t1, hist.history['loss'][0]))
        wandb.log({"train loss":hist.history['loss'][0]}, step = epoch+1)
        wandb.log({"train time (s)": time.time()-t1}, step = epoch+1)
        
        # Evaluation
        if (epoch + 1) % verbose == 0:
            (hits, ndcgs, apks, arks) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, mapk, mark = np.array(hits).mean(), np.array(ndcgs).mean(), np.array(apks).mean(), np.array(arks).mean()

            wandb.log(
                {
                    "HR@{}".format(topK): hr,
                    "NDCG@{}".format(topK): ndcg,
                    "MAP@{}".format(topK): mapk,
                    "MAR@{}".format(topK): mark
                },
                step=epoch + 1,
            )
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                # if args.out > 0:
                model.save_weights(model_out_file, overwrite=True)
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. in [%.2f s]" %(best_iter, best_hr, best_ndcg, time.time()-t))
    #if args.out > 0:
    print("The best MLP model is saved to %s" %(model_out_file))