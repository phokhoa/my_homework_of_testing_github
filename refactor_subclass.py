
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import time
import pymp

import os
from os.path import basename
import pickle

from scipy.signal import convolve2d as conv2
from scipy import signal
from scipy import ndimage

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn import neighbors


# In[2]:


def myKMeans(trn, trn_label, tst, tst_label,num_label,group):
    centroids = []
    for i in range(num_label):
        datai = [x for x, l in zip(trn, trn_label) if l == i]
        centroids.append(np.mean(datai, 0))

    predict = []
    for t in trn:
        dist = []
        for i in range(num_label):
            dist.append(mydistance(t, centroids[i]))
        dist = np.array(dist)
        predict.append(np.argmin(dist))

    trn_acc = np.sum(predict == trn_label) / (1.0 * len(predict))
    trn_acc_unified = np.sum(unify_label(predict, group) == unify_label(trn_label, group)) / (1.0 * len(predict))

    predict = []
    dist = []
    for t in tst:
        d = []
        for i in range(num_label):
            d.append(mydistance(t, centroids[i]))
        d = np.array(d)
        dist.append(d)

    predict = [np.argmin(d) for d in dist]

    tst_acc = np.sum(predict == tst_label) / (1.0 * len(predict))
    tst_acc_unified = np.sum(unify_label(predict, group) == unify_label(tst_label, group)) / (1.0 * len(predict))

    return trn_acc, trn_acc_unified, tst_acc, tst_acc_unified


# In[3]:


def LDA(trn, trn_label, tst, tst_label,num_label,group):
    clf = LinearDiscriminantAnalysis()
    clf.fit(trn, trn_label)

    predict = clf.predict(trn)
    trn_acc = np.sum(predict == trn_label) / (1.0 * len(predict))
    trn_acc_unified = np.sum(unify_label(predict, group) == unify_label(trn_label, group)) / (1.0 * len(predict))

    predict = clf.predict(tst)
    tst_acc = np.sum(predict == tst_label) / (1.0 * len(predict))
    tst_acc_unified = np.sum(unify_label(predict, group) == unify_label(tst_label, group)) / (1.0 * len(predict))

    return trn_acc, trn_acc_unified, tst_acc, tst_acc_unified


# In[4]:


def SVM(trn, trn_label, tst, tst_label,num_label,group):
    clf = svm.LinearSVC()
    clf.fit(trn, trn_label)

    predict = clf.predict(trn)
    trn_acc = np.sum(predict == trn_label) / (1.0 * len(predict))
    trn_acc_unified = np.sum(unify_label(predict, group) == unify_label(trn_label, group)) / (1.0 * len(predict))

    predict = clf.predict(tst)
    tst_acc = np.sum(predict == tst_label) / (1.0 * len(predict))
    tst_acc_unified = np.sum(unify_label(predict, group) == unify_label(tst_label, group)) / (1.0 * len(predict))

    return trn_acc, trn_acc_unified, tst_acc, tst_acc_unified


# In[5]:


def KNN(trn, trn_label, tst, tst_label,num_label,group):
    n_neighbor = 5
    clf = neighbors.KNeighborsClassifier(n_neighbor, weights='distance')
    clf.fit(trn, trn_label)

    predict = clf.predict(trn)
    trn_acc = np.sum(predict == trn_label) / (1.0 * len(predict))
    trn_acc_unified = np.sum(unify_label(predict, group) == unify_label(trn_label, group)) / (1.0 * len(predict))

    predict = clf.predict(tst)
    tst_acc = np.sum(predict == tst_label) / (1.0 * len(predict))
    tst_acc_unified = np.sum(unify_label(predict, group) == unify_label(tst_label, group)) / (1.0 * len(predict))

    return trn_acc, trn_acc_unified, tst_acc, tst_acc_unified


# In[6]:


def myKMeans_top_k(trn, trn_label, tst, tst_label,num_label,group,top_k):
    labels_unified = range(len(group))
    centroids = []
    for i in range(num_label):
        datai = [x for x, l in zip(trn, trn_label) if l == i]
        centroids.append(np.mean(datai, 0))

    predict_probs = []
    for t in trn:
        dist = []
        for i in range(num_label):
            dist.append(mydistance(t, centroids[i]))
        dist = np.array(dist)
        predict_probs.append(dist)
    best_k = np.argsort(predict_probs, axis=1)[:,-top_k:]
    best_k_unified = [unify_label(r,group) for r in best_k]
    best_k_unified = np.array(best_k_unified).tolist()
    prob = [[res.count(l) for l in labels_unified] for res in best_k_unified]
    predict_unified = np.array([np.argmax(p) for p in prob])
    trn_acc_unified = np.sum(predict_unified == unify_label(trn_label, group)) / (1.0 * len(predict_unified))

    predict_probs = []
    for t in tst:
        dist = []
        for i in range(num_label):
            dist.append(mydistance(t, centroids[i]))
        dist = np.array(dist)
        predict_probs.append(dist)
    best_k = np.argsort(predict_probs, axis=1)[:,-top_k:]
    best_k_unified = [unify_label(r,group) for r in best_k]
    best_k_unified = np.array(best_k_unified).tolist()
    prob = [[res.count(l) for l in labels_unified] for res in best_k_unified]
    predict_unified = np.array([np.argmax(p) for p in prob])
    tst_acc_unified = np.sum(predict_unified == unify_label(tst_label, group)) / (1.0 * len(predict_unified))

    return trn_acc_unified,tst_acc_unified


# In[7]:


def LDA_top_k(trn, trn_label, tst, tst_label,num_label,group,top_k):
    labels_unified = range(len(group))
    clf = LinearDiscriminantAnalysis()
    clf.fit(trn, trn_label)

    predict_probs = clf.predict_proba(trn)
    best_k = np.argsort(predict_probs, axis=1)[:,-top_k:]
    best_k_unified = [unify_label(r,group) for r in best_k]
    best_k_unified = np.array(best_k_unified).tolist()
    prob = [[res.count(l) for l in labels_unified] for res in best_k_unified]
    predict_unified = np.array([np.argmax(p) for p in prob])
    trn_acc_unified = np.sum(predict_unified == unify_label(trn_label, group)) / (1.0 * len(predict_unified))

    predict_probs = clf.predict_proba(tst)
    best_k = np.argsort(predict_probs, axis=1)[:,-top_k:]
    best_k_unified = [unify_label(r,group) for r in best_k]
    best_k_unified = np.array(best_k_unified).tolist()
    prob = [[res.count(l) for l in labels_unified] for res in best_k_unified]
    predict_unified = np.array([np.argmax(p) for p in prob])
    tst_acc_unified = np.sum(predict_unified == unify_label(tst_label, group)) / (1.0 * len(predict_unified))

    return trn_acc_unified,tst_acc_unified


# In[8]:


def KNN_top_k(trn, trn_label, tst, tst_label,num_label,group,top_k):
    labels_unified = range(len(group))
    n_neighbor = 5
    clf = neighbors.KNeighborsClassifier(n_neighbor, weights='distance')
    clf.fit(trn, trn_label)

    predict_probs = clf.predict_proba(trn)
    best_k = np.argsort(predict_probs, axis=1)[:,-top_k:]
    best_k_unified = [unify_label(r,group) for r in best_k]
    best_k_unified = np.array(best_k_unified).tolist()
    prob = [[res.count(l) for l in labels_unified] for res in best_k_unified]
    predict_unified = np.array([np.argmax(p) for p in prob])
    trn_acc_unified = np.sum(predict_unified == unify_label(trn_label, group)) / (1.0 * len(predict_unified))

    predict_probs = clf.predict_proba(tst)
    best_k = np.argsort(predict_probs, axis=1)[:,-top_k:]
    best_k_unified = [unify_label(r,group) for r in best_k]
    best_k_unified = np.array(best_k_unified).tolist()
    prob = [[res.count(l) for l in labels_unified] for res in best_k_unified]
    predict_unified = np.array([np.argmax(p) for p in prob])
    tst_acc_unified = np.sum(predict_unified == unify_label(tst_label, group)) / (1.0 * len(predict_unified))

    return trn_acc_unified,tst_acc_unified


# In[9]:


def PCA(data):
    datamean = np.mean(data,0)
    a = data - datamean
    datamax_after_center = np.std(a, 0) + 10e-8
    b = a / datamax_after_center
    x = b.transpose()

    xcov = np.dot(x,x.transpose())/x.shape[1]
    ux,sxx,vx = np.linalg.svd(xcov)

    sx = sxx

    return ux,sx,vx,datamean,datamax_after_center

def customPCA(data, ratio):
    ux,sx,vx, datamean, datamax = PCA(data)

    N = ux.shape[1]
    sum_sx = np.sum(sx)
    ac_sum = 0
    k = 0
    for i in range(N):
        if sx[i] == 0:
            break
        k = i
        ac_sum += sx[i]
        if ac_sum/sum_sx >= ratio:
            break;
    k += 1

    return ux,sx,vx, datamean, datamax, k

def get_pos(fi):
    poses = []
    with open(fi) as f:
        lines = f.readlines()
        for line in lines:
            content = line.split()
            file_dir = content[0]
            tag = os.path.splitext(basename(file_dir))[0]
            num_object = content[1]
            for i in range(int(num_object)):
                x = int(content[2+4*i])
                y = int(content[2+4*i + 1])
                w = int(content[2+4*i + 2])
                h = int(content[2+4*i + 3])
                poses.append([file_dir,x,y,w,h, tag +'_'+ str(i+1)])
    return poses

def filterhighpass(fshift, xsmall, xlarge):
    mask = np.zeros(fshift.shape)
    height = fshift.shape[0]
    width = fshift.shape[1]
    centerx = int(width/2)
    centery = int(height/2)
    mask[centery-xlarge:centery+xlarge,centerx-xlarge:centerx+xlarge] = 1
    mask[centery-xsmall:centery+xsmall,centerx-xsmall:centerx+xsmall] = 0
    fshift_filtered = fshift.copy() * mask
    return fshift_filtered

def unify_label(labels, group):
    res = []
    for a in labels:
        for i in range(len(group)):
            if a in group[i]:
                res.append(i)
    return np.array(res)

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def mydistance(a, b):
    #return np.dot(a-b, (a-b).transpose() )
    return 1 - np.dot(a, b)


# In[10]:


def get_sample(poses,xsmall, xlarge):
    X = []
    for p, l in poses:
        x = p[1]
        y = p[2]
        w = p[3]
        h = p[4]
        image = cv2.imread(p[0].replace("\\", "/"))
        image_preprocess = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        pos = image_preprocess[y:y + h, x:x + w]
        if (w < 2 * xlarge + 1) or (h < 2 * xlarge + 1):
            scale = int(max(2 * xlarge / w, 2 * xlarge / h)) + 1
            pos = cv2.resize(pos, (scale * pos.shape[0], scale * pos.shape[1]))
            h, w = pos.shape

        for deg in range(0, 360, degree_step):
            cur_pos = ndimage.interpolation.rotate(pos, deg)
            h, w = cur_pos.shape
            if (w < 2 * xlarge + 1) or (h < 2 * xlarge + 1):
                scale = int(max(2 * xlarge / w, 2 * xlarge / h)) + 1
                cur_pos = cv2.resize(cur_pos, (scale * cur_pos.shape[0], scale * cur_pos.shape[1]))
                h, w = cur_pos.shape
            f = np.fft.fft2(cur_pos)
            fshift = np.fft.fftshift(f)

            highpass = filterhighpass(fshift, xsmall, xlarge)

            cx = w / 2
            cy = h / 2
            sample = highpass[cy - xlarge:cy + xlarge, cx - xlarge:cx + xlarge]

            # phase = np.arctan2(np.imag(sample), np.real(sample))
            # X.append((phase.reshape(-1),l,pos))

            epsilon = 10 ** -8
            magnitude_spectrum = 20 * np.log(np.abs(sample) + epsilon)
            X.append((magnitude_spectrum.reshape(-1), l, cur_pos))
    return np.array(X)


# In[11]:


pos_files = ['./annotation_subclass/Cyclospora_oocyst_stage1.txt',
             './annotation_subclass/Cyclospora_oocyst_stage2.txt',
             './annotation_subclass/Cyclospora_oocyst_stage3.txt',
             './annotation_subclass/Sarcocystis_oocyst_single_stage2.txt',
             './annotation_subclass/Sarcocystis_oocyst_single_stage3.txt',
             './annotation_subclass/Sarcocystis_oocyst_stage2.txt',
             './annotation_subclass/Sarcocystis_oocyst_stage3.txt',
             './annotation_subclass/Cystoisospora_oocyst_single_stage1.txt',
             './annotation_subclass/Cystoisospora_oocyst_single_stage2.txt',
             './annotation_subclass/Cystoisospora_oocyst_single_stage3.txt',
             './annotation_subclass/Cystoisospora_oocyst_single_stage4.txt',
             './annotation_subclass/Cystoisospora_oocyst_stage1.txt',
             './annotation_subclass/Cystoisospora_oocyst_stage2.txt',
             './annotation_subclass/Cystoisospora_oocyst_stage3.txt',
             './annotation_subclass/Cystoisospora_oocyst_stage4.txt',
             #'./annotation/Acanthamoeba_trophozoite_cyst.txt',
             './annotation/Iodamoeba_cyst.txt',
             './annotation/Toxoplasma_cyst_single.txt',
             #'./annotation/Giardia_trophozoite.txt'
            ]
group = [[0,1,2],[3,4,5,6],[7,8,9,10,11,12,13,14],[15],[16]]
num_label = len(pos_files)


# In[12]:


numlabel = len(pos_files)
raw_poses = []
for i in range(numlabel):
    pfile = pos_files[i]
    raw_poses.append(get_pos(pfile))


# In[13]:


xlarges = [40,50, 60]
pca_keeps = [0.9, 0.95,0.99]
num_k_folds = 20
ratio_test = 0.6
methods = ['myKMeans','LDA','KNN','SVM']
methods_top_k = ['myKMeans_top_k','LDA_top_k','KNN_top_k']
final_res = []

for xlarge in xlarges:
    for pca_keep in pca_keeps:
        foldres = []
        fold_topk_res = []
    	start_time = time.time()
    	with pymp.Parallel(num_k_folds) as p:
        #for fold in range(num_k_folds):
            for fold in p.range(num_k_folds):
                poses_train = []
                poses_test = []
                numtrain = 0
                for i in range(numlabel):
                    poses = raw_poses[i]
                    size = len(poses)
                    train_size = np.minimum(int(np.ceil(ratio_test * size)), 3)
                    if size == 2:
                        train_size = 1
                    numtrain += train_size

                    idx = range(size)
                    np.random.shuffle(idx)
                    for t in idx[:train_size]:
                        poses_train.append((poses[t],i))
                    for t in idx[train_size:]:
                        poses_test.append((poses[t],i))

                poses = poses_train + poses_test
                degree_step = 10
                xsmall = 2


                numtrain = len(poses_train) * (360 / degree_step)
                X = get_sample(poses,xsmall, xlarge)
                X_train = X[:numtrain]
                X_test = X[numtrain:]

                train_samples = np.array([x for x, l, p in X_train])
                train_sample_labels = np.array([l for x, l, p in X_train])
                train_sample_images = np.array([p for x, l, p in X_train])
                train_sample_unified_labels = unify_label(train_sample_labels, group)

                test_samples = np.array([x for x, l, p in X_test])
                test_labels = np.array([l for x, l, p in X_test])
                test_images = np.array([p for x, l, p in X_test])
                test_unified_labels = unify_label(test_labels, group)


                ux, sx, vx, datamean, datamax, k = customPCA(train_samples, pca_keep)
                uxx = ux[:,:k]
                X_train_norm = (train_samples - datamean) / datamax
                X_train_projected = np.dot(X_train_norm, uxx)
                test_samples_norm = (test_samples - datamean) / datamax
                test_samples_projected = np.dot(test_samples_norm, uxx)

                #print pca_keep
                temp_foldres = []
                for method in methods:
                    trn_acc, trn_acc_unified, tst_acc, tst_acc_unified = eval(method)(X_train_projected, train_sample_labels,
                                                                              test_samples_projected, test_labels,
                                                                              num_label,group)
                    #print trn_acc, trn_acc_unified, tst_acc, tst_acc_unified
                    temp_foldres.append([trn_acc, trn_acc_unified, tst_acc, tst_acc_unified])
                foldres.append(temp_foldres)
                temp_fold_topk_res = []
                for method in methods_top_k:
                    trn_acc_unified, tst_acc_unified = eval(method)(X_train_projected, train_sample_labels,
                                                                              test_samples_projected, test_labels,
                                                                              num_label,group, 3)
                    #print trn_acc_unified, tst_acc_unified
                    temp_fold_topk_res.append([trn_acc_unified, tst_acc_unified])
                fold_topk_res.append(temp_fold_topk_res)
                print '%d/%d'%(fold, num_k_folds)
        ave_foldres = np.mean(foldres, axis=0)
        ave_fold_topk_res = np.mean(fold_topk_res, axis=0)
        #print [xlarge,pca_keep,ave_foldres,ave_fold_topk_res,foldres,fold_topk_res]
        #final_res.append([xlarge,pca_keep,ave_foldres,ave_fold_topk_res,foldres,fold_topk_res])
        print [xlarge,pca_keep,ave_foldres,ave_fold_topk_res]
        final_res.append([xlarge,pca_keep,ave_foldres,ave_fold_topk_res])
	print("--- %s seconds ---" % (time.time() - start_time))



# In[14]:


with open('outfile','wb') as fp:
    pickle.dump(final_res,fp)
with open('outfile', 'rb') as fp:
    load_res = pickle.load(fp)

print load_res


# methods = ['myKMeans']
# for mt in methods:
#     trn_acc, trn_acc_unified, tst_acc, tst_acc_unified = eval(mt)(X_train_projected, train_sample_labels,
#                                                                           test_samples_projected, test_labels,
#                                                                           num_label,group)
#     print pca_keep
#     print trn_acc, trn_acc_unified, tst_acc, tst_acc_unified
