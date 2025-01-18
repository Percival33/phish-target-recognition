import tensorflow as tf 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten


from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Conv2D,MaxPooling2D,Input,Lambda,GlobalMaxPooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.applications.vgg16 import VGG16

from matplotlib.pyplot import imread
from skimage.transform import rescale, resize
from skimage.io import imsave

import os
import numpy as np
from dataclasses import dataclass

################################## read_imgs_per_website ##################################
def read_imgs_per_website(data_path,targets,imgs_num,reshape_size,start_target_count):
    all_imgs = np.zeros(shape=[imgs_num,224,224,3])
    all_labels = np.zeros(shape=[imgs_num,1])
    
    all_file_names = []
    targets_list = targets.splitlines()
    count = 0
    for i in range(0,len(targets_list)):
        target_path = data_path + targets_list[i]
        print(target_path)
        file_names = sorted(os.listdir(target_path))
        for j in range(0,len(file_names)):
            try:
                img = imread(target_path+'/'+file_names[j])
                img = img[:,:,0:3]
                all_imgs[count,:,:,:] = resize(img, (reshape_size[0], reshape_size[1]),anti_aliasing=True)
                all_labels[count,:] = i + start_target_count
                all_file_names.append(file_names[j])
                count = count + 1
            except:
                #some images were saved with a wrong extensions 
                try:
                    img = imread(target_path+'/'+file_names[j],format='jpeg')
                    img = img[:,:,0:3]
                    all_imgs[count,:,:,:] = resize(img, (reshape_size[0], reshape_size[1]),anti_aliasing=True)
                    all_labels[count,:] = i + start_target_count
                    all_file_names.append(file_names[j])
                    count = count + 1
                except:
                    print('failed at:')
                    print('***')
                    print(file_names[j])
                    break 
    return all_imgs,all_labels,all_file_names

################################## Order and label targets ##################################
def order_random_array(orig_arr,y_orig_arr,targets):
    sorted_arr = np.zeros(orig_arr.shape)
    y_sorted_arr = np.zeros(y_orig_arr.shape)
    count = 0
    for i in range(0,targets):
        for j in range(0,orig_arr.shape[0]):
            if y_orig_arr[j] == i:
                sorted_arr[count,:,:,:] = orig_arr[j,:,:,:]
                y_sorted_arr[count,:] = i
                count = count + 1
    return sorted_arr,y_sorted_arr 

# Store the start and end of each target in the phishing set (used later in triplet sampling)
# Not all targets might be in the phishing set 
def start_end_each_target(num_target,labels):
    prev_target = 0
    start_end_each_target = np.zeros((num_target,2))
    start_end_each_target[0,0] = 0
    count_target = 0
    for i in range(1,labels.shape[0]):
        if not labels[i] == prev_target:
            start_end_each_target[count_target,1] = i-1
            count_target = count_target + 1
            start_end_each_target[count_target,0] = i
            prev_target = prev_target + 1
    start_end_each_target[num_target-1,1] = labels.shape[0]-1
    return start_end_each_target


# Store the start and end of each target in the training set (used later in triplet sampling)
def all_targets_start_end(num_target,labels):
    prev_target = labels[0]
    start_end_each_target = np.zeros((num_target,2))
    start_end_each_target[0,0] = labels[0]
    if not labels[0] == 0:
        start_end_each_target[0,0] = -1
        start_end_each_target[0,1] = -1
    count_target = 0
    for i in range(1,labels.shape[0]):
        if not labels[i] == prev_target:
            start_end_each_target[int(labels[i-1]),1] = int(i-1)
            #count_target = count_target + 1
            start_end_each_target[int(labels[i]),0] = int(i)
            prev_target = labels[i]
    start_end_each_target[int(labels[-1]),1] = int(labels.shape[0]-1)
    
    for i in range(1,num_target):
        if start_end_each_target[i,0] == 0:
            print(i)
            start_end_each_target[i,0] = -1
            start_end_each_target[i,1] = -1
    return start_end_each_target

################################## Finding Hard subsets ##################################
# Find a query set for each target
def find_fixed_set_idx(labels_start_end_train_legit, num_target):
    website_random_idx = np.zeros([num_target,])
    for i in range(0,num_target):
        class_idx_start_end = labels_start_end_train_legit[i,:]
        website_random_idx[i] = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
    return website_random_idx

# Compute L2 distance between embeddings
def compute_distance_pair(layer1,layer2):
    diff = layer1 - layer2
    l2_diff = np.mean(diff**2)
    return l2_diff
    
# Compute the embeddings of the query set, the phishing training set, the training whitelist 
def predict_all_imgs(model):
    X_train_legit_last_layer = model.predict(X_train_legit,batch_size=10)
    X_train_phish_last_layer = model.predict(X_train_phish,batch_size=10)
    fixed_set_last_layer = model.predict(fixed_set,batch_size=10)
    
    return X_train_legit_last_layer,X_train_phish_last_layer,fixed_set_last_layer

# Compute distance between the query set and all training examples 
def compute_all_distances(fixed_set,train_legit,train_phish):
    train_size = train_legit.shape[0] + train_phish.shape[0]
    X_all_train = np.concatenate((train_legit,train_phish))
    pairwise_distance = np.zeros([fixed_set.shape[0],train_size])
    for i in range(0,fixed_set.shape[0]):
        pair1 = fixed_set[i,:]
        for j in range(0,train_size):
            pair2 = X_all_train[j,:]
            l2_diff = compute_distance_pair(pair1,pair2)
            pairwise_distance[i,j] = l2_diff
    return pairwise_distance

# Get index of false positives (different-website examples with small distance) of one query image
def find_n_false_positives(distances,n,test_label):
    count = 0
    X_false_pos_idx = np.zeros([n,])
    idx_min = np.argsort(distances)
    for i in range(0,distances.shape[0]):
        next_min_idx = idx_min[i]
        n_label = y_train[next_min_idx]
        #false positives (have close distance even if they are from differenet category)
        if not (test_label == n_label):
            X_false_pos_idx[count] = next_min_idx
            count = count + 1
            if count == n:
                break 
    while count < n:
        idx_min[count] = -1
        count = count + 1
    return X_false_pos_idx

# Get index of false negatives (same-website examples with large distance) of one query image
def find_n_false_negatives(distances,n,test_label):
    count = 0     
    X_false_neg_idx = np.zeros([n,])
    idx_max = np.argsort(distances)[::-1]
    for i in range(0,distances.shape[0]):
        next_max_idx = idx_max[i]
        n_label = y_train[next_max_idx]
        #false negatives (have large distance although they are in the same category )
        if test_label == n_label:
            X_false_neg_idx[count] = next_max_idx
            count = count + 1
            if count == n:
                break
    while count < n:
        idx_max[count] = -1
        count = count + 1
    return X_false_neg_idx

# Get the idx of false positives and false negtaives for all query examples
def find_index_for_all_set(distances,n):
    all_idx = np.zeros([distances.shape[0],2,n])
    for i in range(0,distances.shape[0]):
        distance_i = distances[i,:]
        all_idx[i,0,:] = find_n_false_positives(distance_i,n,i)
        all_idx[i,1,:] = find_n_false_negatives(distance_i,n,i)
    return all_idx

# Form the new training set based on the hard examples indices of all query images
def find_next_training_set(all_idx,n):
    global X_train_new, y_train_new
    all_idx = all_idx.astype(int)
    count = 0
    for i in range(all_idx.shape[0]):
        for j in range(0,n):
            if not all_idx[i,0,j] == -1:
                X_train_new[count,:,:,:] = X_train[all_idx[i,0,j],:,:,:]
                y_train_new[count,:] = y_train[all_idx[i,0,j]]
                count = count +1 
        for j in range(0,n):
            if not all_idx[i,1,j] == -1:
                X_train_new[count,:,:,:] = X_train[all_idx[i,1,j],:,:,:]
                y_train_new[count,:] = y_train[all_idx[i,1,j]]
                count = count +1 
    X_train_new = X_train_new[0:count,:]
    y_train_new = y_train_new[0:count,:]
    return X_train_new,y_train_new

# Main function for subset sampling
# Steps:
    # Predict all images
    # Find pairwise distances between query and training set
    # Find indices of hard positive and negative examples
    # Find new training set
    # Order training set by targets
def find_main_train(model,fixed_set,targets):
    X_train_legit_last_layer,X_train_phish_last_layer,fixed_set_last_layer = predict_all_imgs(model)
    pairwise_distance = compute_all_distances(fixed_set_last_layer,X_train_legit_last_layer,X_train_phish_last_layer)
    n = 1
    all_idx = find_index_for_all_set(pairwise_distance,n)
    X_train_new,y_train_new = find_next_training_set(all_idx,n)
    X_train_new,y_train_new = order_random_array(X_train_new,y_train_new,targets)
    labels_start_end_train = start_end_each_target(targets,y_train_new)
    return X_train_new,y_train_new,labels_start_end_train

def get_idx_of_target(target_name,all_targets):
    for i in range(0,len(all_targets)):
        if all_targets[i] == target_name:
            found_idx = i
            return found_idx
        
def get_associated_targets_idx(target_lists,all_targets):
    sub_target_lists_idx = []
    parents_ids = []
    for i in range(0,len(target_lists)):
        target_list = target_lists[i]
        parent_target = target_list[0]
        one_target_list = []
        parent_idx = get_idx_of_target(parent_target,all_targets)
        parents_ids.append(parent_idx)
        for child_target in target_list[1:]:
            child_idx = get_idx_of_target(child_target,all_targets)
            one_target_list.append(child_idx)
        sub_target_lists_idx.append(one_target_list)
    return parents_ids,sub_target_lists_idx 

def check_if_same_category(img_label1,img_label2):
    if_same = 0
    if img_label1 in parents_ids:
        if img_label2 in sub_target_lists_idx[parents_ids.index(img_label1)]:
            if_same = 1
    elif img_label1 in sub_target_lists_idx[0]:
        if img_label2 in sub_target_lists_idx[0] or img_label2 == parents_ids[0]:
            if_same = 1
    elif img_label1 in sub_target_lists_idx[1]:
        if img_label2 in sub_target_lists_idx[1] or img_label2 == parents_ids[1]:
            if_same = 1
    elif img_label1 in sub_target_lists_idx[2]:
        if img_label2 in sub_target_lists_idx[2] or img_label2 == parents_ids[2]:
            if_same = 1
    return if_same

################################## Triplet Sampling ##################################
def pick_first_img_idx(labels_start_end,num_targets):
    random_target = -1
    while (random_target == -1):
        random_target = np.random.randint(low = 0,high = num_targets)
        if labels_start_end[random_target,0] == -1:
            random_target = -1
    return random_target

def pick_pos_img_idx(img_label):
    class_idx_start_end = labels_start_end_train[img_label,:]
    same_idx = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
    img = X_train_new[same_idx,:]
    return img

def pick_neg_img(anchor_idx,num_targets):
    if anchor_idx == 0:
        targets = np.arange(1,num_targets)
    elif anchor_idx == num_targets -1:
        targets = np.arange(0,num_targets-1)
    else:
        targets = np.concatenate([np.arange(0,anchor_idx),np.arange(anchor_idx+1,num_targets)])
    diff_target_idx = np.random.randint(low = 0,high = num_targets-1)
    diff_target = targets[diff_target_idx]
    
    class_idx_start_end = labels_start_end_train[diff_target,:]
    idx_from_diff_target = np.random.randint(low = class_idx_start_end[0],high = class_idx_start_end[1]+1)
    img = X_train_new[idx_from_diff_target,:]
    
    return img,diff_target

#Sample batch 
def get_batch(batch_size,train_fixed_set,num_targets):
   
    # initialize 3 empty arrays for the input image batch
    h = X_train_legit.shape[1]
    w = X_train_legit.shape[2]
    triple=[np.zeros((batch_size, h, w,3)) for i in range(3)]

    for i in range(0,batch_size):
        img_idx_pair1 = pick_first_img_idx(labels_start_end_train,num_targets)
        triple[0][i,:,:,:] = train_fixed_set[img_idx_pair1,:]
        img_label = img_idx_pair1
        
        #get image for the second: positive
        triple[1][i,:,:,:] = pick_pos_img_idx(img_label)
            
        #get image for the thrid: negative from legit
        img_neg,label_neg = pick_neg_img(img_label,num_targets)
        while check_if_same_category(img_label,label_neg) == 1:
            img_neg,label_neg = pick_neg_img(img_label,num_targets)

        triple[2][i,:,:,:] = img_neg
          
    return triple

################################## Load Model ##################################
def loss(y_true,y_pred):
    loss_value = K.maximum(y_true, margin + y_pred)
    loss_value = K.mean(loss_value,axis=0)
    return loss_value

def custom_loss(margin):
    def loss(y_true,y_pred):
        loss_value = K.maximum(y_true, margin + y_pred)
        loss_value = K.mean(loss_value,axis=0)
        return loss_value
    return loss

################################## Training ##################################
def save_keras_model(model):
    model.save(output_dir+new_saved_model_name+'.h5')
    print("Saved model to disk")

    


"""
output_dir = '../../../notebooks/'
dataset_path = '../../../data/processed/smallerSampleDataset/'

phish_emb_name = 'phishing_emb.npy'
phish_emb_labels_name = 'phishing_labels.npy'

phish_train_idx_name = 'train_idx.npy'
phish_test_idx_name = 'test_idx.npy'

train_emb_name = 'whitelist_emb.npy'
train_emb_labels_name = 'whitelist_labels.npy'

#precomputed attacks embeddings for the phishing test set if any. 
#set use_attack to 1 to compute based on this
phish_emb_test_attack = 'X_phish_test_noise_gamma.npy'
use_attack = 0

X_legit_train = np.load(output_dir+train_emb_name)
y_legit_train = np.load(output_dir+train_emb_labels_name)

X_phish = np.load(output_dir+phish_emb_name)
y_phish = np.load(output_dir+phish_emb_labels_name)

phish_test_idx = np.load(output_dir+phish_test_idx_name)
phish_train_idx = np.load(output_dir+phish_train_idx_name)

X_phish_test = X_phish[phish_test_idx,:]
y_phish_test = y_phish[phish_test_idx,:]

#set the phishing test set directly to the precomputed embeddings of the attack
if use_attack == 1:
    X_phish_test = np.load(output_dir+phish_emb_test_attack)
    print('Test on: '+phish_emb_test_attack)

X_phish_train = X_phish[phish_train_idx,:]
y_phish_train = y_phish[phish_train_idx,:]
"""

@dataclass
class DataSet:
    phish_test_idx: np.array
    phish_train_idx: np.array
    legit_file_names: list[str]
    phish_file_names: list[str]

    X_legit_train: np.array
    y_legit_train: np.array
    X_phish: np.array
    y_phish: np.array

    X_phish_test: np.array
    y_phish_test: np.array

    X_phish_train = X_phish[phish_train_idx,:]
    y_phish_train = y_phish[phish_train_idx,:]

class Evaluate:
    # Find same-category website (matching is correct if it was matched to the same category (e.g. microsoft and outlook ))
    parents_targets = ['microsoft','apple','google','alibaba']
    sub_targets = [['ms_outlook','ms_office','ms_bing','ms_onedrive','ms_skype'],['itunes','icloud'],['google_drive'],['aliexpress']]

    parents_targets_idx = [90,12,65,4]
    sub_targets = [[150,152,151,149,148],[153,154],[147],[5]]


    def __init__(self, dataset):
        self.X_phish_train = dataset.X_phish_train
        self.X_phish_test = dataset.X_phish_test
        self.phish_train_idx = dataset.phish_train_idx
        self.X_legit_train = dataset.X_legit_train

        self.legit_file_names = dataset.legit_file_names
        self.phish_file_names = dataset.phish_file_names

        self.pairwise_distance = self.compute_all_distances(self.X_phish_test)


    # L2 distance
    def compute_distance_pair(self, layer1,layer2):
        diff = layer1 - layer2
        l2_diff = np.sum(diff**2) / self.X_phish_train.shape[1]
        return l2_diff

    # Pairwise distance between query image and training
    def compute_all_distances(self, test_matrix):
        train_size = self.phish_train_idx.shape[0] + self.X_legit_train.shape[0]
        X_all_train = np.concatenate((self.X_phish_train,self.X_legit_train))
        pairwise_distance = np.zeros([test_matrix.shape[0],train_size])
        for i in range(0,test_matrix.shape[0]):
            pair1 = test_matrix[i,:]
            for j in range(0,train_size):
                pair2 = X_all_train[j,:]
                l2_diff = self.compute_distance_pair(pair1,pair2)
                pairwise_distance[i,j] = l2_diff
        return pairwise_distance

    # Find Smallest n distances
    def find_min_distances(distances,n):
        idx = distances.argsort()[:n]
        values = distances[idx]
        return idx,values

    # Find names of examples with min distance
    def find_names_min_distances(self,idx,values):
        names_min_distance = ''
        only_names = []
        distances = ''
        for i in range(0,idx.shape[0]):
            index_min_distance = idx[i]
            if (index_min_distance < self.X_phish_train.shape[0]):
                names_min_distance = names_min_distance + 'Phish: ' + self.phish_train_file_names[index_min_distance] +','
                only_names.append(self.phish_train_file_names[index_min_distance])   
            else:
                names_min_distance = names_min_distance + 'Legit: ' + self.legit_file_names[index_min_distance-self.X_phish_train.shape[0]] +','
                only_names.append(self.legit_file_names[index_min_distance-self.X_phish_train.shape[0]])   
            distances = distances + str(values[i]) + ','
        names_min_distance = names_min_distance[:-1]
        distances = distances[:-1]
        return names_min_distance,only_names,distances

    # this function maps sub targets if they were split
    def check_if_same_category(self,img_label1,img_label2):
        if_same = 0
        if img_label1 in self.parents_targets_idx:
            if img_label2 in self.sub_targets[self.parents_targets_idx.index(img_label1)]:
                if_same = 1
        elif img_label1 in self.sub_targets[0]:
            if img_label2 in self.sub_targets[0] or img_label2 == self.parents_targets_idx[0]:
                if_same = 1
        elif img_label1 in self.sub_targets[1]:
            if img_label2 in self.sub_targets[1] or img_label2 == self.parents_targets_idx[1]:
                if_same = 1
        elif img_label1 in self.sub_targets[2]:
            if img_label2 in self.sub_targets[2] or img_label2 == self.parents_targets_idx[2]:
                if_same = 1
        return if_same

    # Find if target is in the top closest n distances
    def check_if_target_in_top(self, test_file_name,only_names):
        found = 0
        idx = 0
        test_label = self.get_label_from_name(test_file_name)
        print('***')
        print('Test example: '+test_file_name)
        for i in range(0,len(only_names)):
            label_distance = self.get_label_from_name(only_names[i])
            if label_distance == test_label or check_if_same_category(test_label,label_distance) == 1:
                found = 1
                idx = i+1
                print('found')
                break
        return found,idx