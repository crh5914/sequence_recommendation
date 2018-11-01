import tensorflow as tf
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='run single embedding model')
    parser.add_argument('--num_factors',type=int,default=64,help='item embedding dimension')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--epochs',type=int,default=50,help='trainning epochs')
    parser.add_argument('--dataset',type=str,default='ml100k',help='specify the trainning dataset')
    paser.add_argument('--sep',type=str,default='\t',help='specify the dilimiter')
    return parser.parse_args()

