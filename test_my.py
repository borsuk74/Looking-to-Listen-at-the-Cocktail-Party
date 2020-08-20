import sys
sys.path.append('./ model / model')
sys.path.append('./ model / utils')
from keras.models import load_model

import os
import scipy.io.wavfile as wavfile
import numpy as np
#import utils
import tensorflow as tf
from model.utils import utils
import time

#parameters
people = 2
num_gpu=1

#path
model_path = './saved_AV_models/AVmodel-2p-020-0.51971.h5'
result_path = './predict/'
os.makedirs(result_path,exist_ok=True)

database = './data/AV_model_database/mix/'
face_emb_path = './model/face_embedding/face1022_emb/'
print('Initialing Parameters......')




def get_data_name(line,people=people,database=database,face_emb=face_emb_path):
    parts = line.split() # get each name of file for one testset
    mix_str = parts[0]
    name_list = mix_str.replace('.npy','')
    name_list = name_list.replace('mix-','',1)
    names = name_list.split('-')
    single_idxs = []
    for i in range(people):
        single_idxs.append(names[i])
    file_path = database + mix_str
    mix = np.load(file_path)
    face_embs = np.zeros((1,75,1,1792,people))
    for i in range(people):
        face_embs[0,:,:,:,i] = np.load(face_emb+single_idxs[i]+"_face_emb.npy")

    return mix,single_idxs,face_embs

#loading data
print('Loading data ......')
test_file = []
with open('./data/AVdataset_val.txt','r') as f:
    test_file = f.readlines()

from model.model.loss import audio_discriminate_loss2 as audio_loss
loss = audio_loss(gamma=0.1, beta=0.2, people_num=people)

av_model = load_model(model_path,custom_objects={'tf':tf,'loss_func': loss})

for line in test_file[18:19]:
        mix,single_idxs,face_embed = get_data_name(line,people,database,face_emb_path)
        mix_ex = np.expand_dims(mix,axis=0)
        start = time.time()
        cRMs = av_model.predict([mix_ex,face_embed])
        print(time.time() -start)
        cRMs = cRMs[0]
        prefix =''
        for idx in single_idxs:
            prefix +=idx+'-'
        for i in range(cRMs.shape[-1]):
            cRM =cRMs[:,:,:,i]
            assert cRM.shape ==(298,257,2)
            F = utils.fast_icRM(mix,cRM)
            print(F.shape)
            T = utils.fast_istft(F,power=False)#default was false
            filename = result_path+str(single_idxs[i])+'.wav'
            wavfile.write(filename,16000,T)
