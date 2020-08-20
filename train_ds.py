import tensorflow as tf

import sys

sys.path.append('model/utils')
sys.path.append('data/audio_data')
import os
import librosa
import numpy as np
import utils
import audio_data_ds
import itertools
import functools
import time
import random
import math
import re
import scipy.io.wavfile as wavfile

sys.path.append('./model/model/')
import AV_model as AV
from option import latest_file
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import multi_gpu_model
#from data_load import AVGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import os
from loss import audio_discriminate_loss2 as audio_loss


data_range = (0, 184)  # data usage to generate database
audio_norm_path = os.path.expanduser("./data/audio_data/norm_audio_train")
# database_path = '../AV_model_database'
frame_valid_path = './data/video_data/valid_face_text.txt'
face_embed_path = './model/face_embedding/face1022_emb/'
single_dir_path = './data/AV_model_database/single/'
num_speakers = 2
max_generate_data = 25000
sampling_rate = 16000

def transform_fn(file_names, face_path=face_embed_path, single_path=single_dir_path, sr=16000):
    '''This function will generate required mix and crm tensors dynamically'''
    # from input generate stft of mixed signal X1
    number_of_speakers = len(file_names)
    mix_rate = 1.0 / float(number_of_speakers)
    wav_list = []
    faces_list = []
    idx_list = []
    for part_idx in range(number_of_speakers):
        path = file_names[part_idx]#.numpy().decode('utf8')
        file_name = path.split('/')[-1]
        idx = int(re.findall("\d+", file_name)[0])
        face_file_name = '{:05d}_face_emb.npy'.format(idx)
        wav, _ = librosa.load(path, sr=sr)
        wav_list.append(wav)
        faces_list.append(face_file_name)
        idx_list.append(idx)

    mix_wav = np.zeros_like(wav_list[0])
    for wav in wav_list:
        mix_wav += wav * mix_rate
    F_mix = utils.fast_stft(mix_wav)
    #read files with face embeddings and compute cRMs for input channels
    X2dim = (75, 1, 1792, number_of_speakers)
    X2 = np.empty(X2dim)
    y_dim = (298, 257, 2, 2)
    y = np.empty(y_dim)
    for j in range(number_of_speakers):
        X2[:, :, :, j] = np.load(face_path + faces_list[j])
        single_name = 'single-{:05d}.npy'.format(idx_list[j])
        F_single = np.load(single_path + single_name)
        cRM = utils.fast_cRM(F_single, F_mix)
        y[:, :, :, j] = cRM

    return ((F_mix, X2), y)

# Need to make sure that  /single/.npy files are present for valid files in range by running audio_data_ds.py file
def get_generator(data_range, audio_norm_path, frame_valid_path, num_speakers=2, max_data=25000):
    # pick up valid audio files within specified range
    audio_path_list = audio_data_ds.generate_data_list(data_r=data_range,
                                                    audio_norm_pth=audio_norm_path,
                                                    frame_valid=frame_valid_path)
    # populate split_list according to the number of speakers
    length = len(audio_path_list)
    part_len = length // num_speakers
    start = 0
    split_list = []
    while (start + part_len) <= length:
        part = audio_path_list[start:(start + part_len)]
        split_list.append(part)
        start += part_len

    # Generate all possible permutations of the indexes
    assert len(split_list) == num_speakers
    part_len = len(split_list[-1])
    idx_list = [i for i in range(part_len)]
    combo_idx_list = itertools.product(idx_list, repeat=num_speakers)
    #ret_list = []
    for combo_idx in combo_idx_list:
        assert len(combo_idx) == len(split_list)
        file_names = [split_list[part_idx][combo_idx[part_idx]][1] for part_idx in range(len(split_list))]
        yield transform_fn(file_names, face_path=face_embed_path, single_path=single_dir_path, sr=16000)



gen = functools.partial(get_generator,  data_range=data_range,
                                                      audio_norm_path=audio_norm_path,
                                                      frame_valid_path=frame_valid_path,
                                                   num_speakers=2)

ds = tf.data.Dataset.from_generator(gen, ((tf.float32, tf.float32), tf.float32),
                                    ((tf.TensorShape([ 298, 257, 2]), tf.TensorShape([ 75, 1, 1792, 2])),
                                     tf.TensorShape([ 298, 257, 2, 2])))


# Resume Model
resume_state = False

# Parameters
people_num = 2
epochs = 20
initial_epoch = 0
batch_size = 2
gamma_loss = 0.1
beta_loss = gamma_loss * 2



ds = ds.shuffle(buffer_size=100)
ds = ds.batch(batch_size=batch_size)
ds = ds.prefetch(buffer_size=1)

# PATH
model_path = './saved_AV_models'  # model path
database_path = 'data/'

# create folder to save models
folder = os.path.exists(model_path)
if not folder:
    os.makedirs(model_path)
    print('create folder to save models')
filepath = model_path + "/AVmodel-" + str(people_num) + "p-{epoch:03d}-{loss:.5f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


# automatically change lr
def scheduler(epoch):
    ini_lr = 0.00001
    lr = ini_lr
    if epoch >= 5:
        lr = ini_lr / 5
    if epoch >= 10:
        lr = ini_lr / 10
    return lr

rlr = LearningRateScheduler(scheduler, verbose=1)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    AV_model = None
    # the training steps
    if resume_state:
        latest_file = latest_file(model_path + '/')
        AV_model = load_model(latest_file, custom_objects={"tf": tf})
        info = latest_file.strip().split('-')
        initial_epoch = int(info[-2])
    else:
        AV_model = AV.AV_model(people_num)

    adam = optimizers.Adam()
    loss = audio_loss(gamma=gamma_loss, beta=beta_loss, people_num=people_num)
    AV_model.compile(loss=loss, optimizer=adam)
    print(AV_model.summary())

    history = AV_model.fit(ds, epochs=epochs,
                                           callbacks=[TensorBoard(log_dir='./log_AV'), checkpoint, rlr],
                                           initial_epoch=initial_epoch)

