
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
from os import listdir
from os.path import basename,isfile,join
import cPickle as pickle
#from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn import neighbors
from sklearn import svm
from skimage.feature import hog
from skimage import feature
import random
import time



# In[2]:


def creat_posfile_group(pos_files_group):
    i = 0
    pos_files = []
    group = []
    for pfs in pos_files_group:
        group_idx = []
        for j in range(len(pfs)):
            group_idx.append(i)
            i = i+1
            pos_files.append(pfs[j])
        group.append(group_idx)

    return pos_files,group

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

def get_neg_images(fi):
    neg_images = []
    with open(fi) as f:
        lines = f.readlines()
        for line in lines:
            content = line.split()
            file_dir = content[0]
            image = cv2.imread(file_dir.replace("\\", "/"))
            num_object = content[1]
            for i in range(int(num_object)):
                x = int(content[2+4*i])
                y = int(content[2+4*i + 1])
                w = int(content[2+4*i + 2])
                h = int(content[2+4*i + 3])
                image[y:y+h, x:x+w] = 0
            neg_images.append(image)
    return neg_images


def unify_label(labels, group):
    res = []
    for a in labels:
        for i in range(len(group)):
            if a in group[i]:
                res.append(i)
    return np.array(res)

def highpass_and_imgback(pos,xsmall, xlarge):
    cur_h, cur_w = pos.shape
    if cur_h < 2*xlarge or cur_w < 2*xlarge:
        #print cur_h, cur_w
        resize_scale = int(max(2 * xlarge / (1.0*cur_w), 2 * xlarge / (cur_h*1.0))) + 1
        #print resize_scale
        pos = cv2.resize(pos,(0,0),fx = resize_scale,fy=resize_scale)
        cur_h, cur_w = pos.shape
        #print cur_h, cur_w


    f = np.fft.fft2(pos)
    fshift = np.fft.fftshift(f)



    mask = np.zeros(pos.shape)
    #mask = np.ones(pos.shape)

    height = pos.shape[0]
    width = pos.shape[1]
    centerx = int(width/2)
    centery = int(height/2)

    mask[centery-xlarge:centery+xlarge,centerx-xlarge:centerx+xlarge] = 1
    mask[centery-xsmall:centery+xsmall,centerx-xsmall:centerx+xsmall] = 0
    fshift_filtered = fshift.copy() * mask
    f_ishift_filtered = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift_filtered)

    mag_fshift = fshift_filtered[centery-xlarge:centery+xlarge,centerx-xlarge:centerx+xlarge]
    epsilon = 10**-8
    magnitude_spectrum = 20*np.log(np.abs(mag_fshift) + epsilon)

    img_back = np.abs(img_back)
    min_back = np.min(img_back)
    max_back = np.max(img_back)

    img_back = np.array((img_back-min_back)/(max_back-min_back)*255,dtype = np.uint8)
    img_back = np.array(img_back, dtype = np.uint8)
    img_back = np.clip(img_back, 0, 255)

    return img_back, magnitude_spectrum

def get_sample(poses,xsmall, xlarge):
    X = []
    for p, l in poses:
        x = p[1]
        y = p[2]
        w = p[3]
        h = p[4]
        image = cv2.imread(p[0].replace("\\", "/"))
        image_preprocess = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image_preprocess = highpass_and_imgback(image_preprocess,xsmall, xlarge)


        pos = image_preprocess[y:y + h, x:x + w]
        """
        if (w < 2 * xlarge + 1) or (h < 2 * xlarge + 1):
            scale = int(max(2 * xlarge / w, 2 * xlarge / h)) + 1
            pos = cv2.resize(pos, (scale * pos.shape[0], scale * pos.shape[1]))
            #print pos.shape
            #h, w = pos.shape

        #pos_back, magnitude = highpass_and_imgback(pos,xsmall, xlarge)
        #pos_back = cv2.resize(pos_back, (h,w))

        deg_step = 10
        for deg in range(0,360, deg_step):
            cur_pos = ndimage.interpolation.rotate(pos,deg, reshape=True)
            cur_pos_back, magnitude = highpass_and_imgback(cur_pos,xsmall, xlarge)
            X.append((magnitude,l,cur_pos))
        """
        deg_step = 10
        for deg in range(0,360, deg_step):
            cur_pos = ndimage.interpolation.rotate(pos,deg, reshape=True)
            cur_h, cur_w = cur_pos.shape
            new_pos = cur_pos
            new_pos = np.array(new_pos, np.uint8)
            if (cur_w < 2 * xlarge + 1) or (cur_h < 2 * xlarge + 1):
                scale = int(max(2 * xlarge / cur_w, 2 * xlarge / cur_h)) + 1
                new_pos = cv2.resize(cur_pos, (scale * cur_pos.shape[1], scale * cur_pos.shape[0]))
                new_pos = np.array(new_pos, np.uint8)

            pos_back, magnitude = highpass_and_imgback(new_pos,xsmall, xlarge)
            X.append((magnitude,l,new_pos))




        #X.append((pos_back,l,pos))
    return X




def PCA(data):
    datamean = np.mean(data,0)
    a = data - datamean
    datamax_after_center = np.std(a,0) + 10e-8
    b = a / datamax_after_center
    x = b.transpose()

    xcov = np.dot(x,x.transpose())/x.shape[1]
    ux,sxx,vx = np.linalg.svd(xcov)

    sx = sxx
    return ux,sx,vx, datamean, datamax_after_center

def customPCA(data, ratio):
    ux,sx,vx, datamean, datamax = PCA(data)

    N = ux.shape[1]
    sum_sx = np.sum(sx)
    ac_sum = 0
    k = 0
    for _i in range(N):
        if sx[_i] == 0:
            break
        k = _i
        ac_sum += sx[_i]
        if ac_sum/sum_sx >= ratio:
            break
    k += 1
    return ux,sx,vx,datamean,datamax,k

def mydistance(a, b):
    #return np.dot(a-b, (a-b).transpose() )
    #return 1 - np.dot(a, b)
    return 1 - np.dot(a, b)/(np.sqrt(np.dot(a,a)*np.dot(b,b)))
    #return (1 - np.dot(a, b)/(np.sqrt(np.dot(a,a)*np.dot(b,b))))**2 + (1 - (np.sqrt(np.dot(a,a)/np.dot(b,b))))**2


# In[3]:


def check_foreground_hog(hypothesis, clf, _sz):
    cur_pos = hypothesis
    cur_h, cur_w = cur_pos.shape
    if (cur_w < int(_sz[1])) or (cur_h < int(_sz[0])):
        cur_pos = cv2.resize(cur_pos,(int(_sz[1]),int(_sz[0])))

    hogFeature = feature.hog(cur_pos,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          transform_sqrt=True,
                          visualise=True,
                          feature_vector=False)[0]
    predict = clf.predict([hogFeature.reshape(-1)])[0]
    if predict == 1:
        return True
    else:
        return False


# In[ ]:

def reconstruct_from_pca(x,ux,sx,vx,datamean, datamax,k):
    N = ux.shape[1]
    padded = np.zeros((N))
    padded[:k] = x

    reconstructed = np.dot(ux,padded)
    return reconstructed
def check_foreground(hypothesis, max_error,min_norm, xsmall, xlarge,ux,sx,vx,datamean, datamax,k):
    uxx = ux[:,:k]

    cur_pos = hypothesis
    cur_h, cur_w = cur_pos.shape
    if (cur_w < 2 * xlarge + 1) or (cur_h < 2 * xlarge + 1):
                scale = int(max(2 * xlarge / cur_w, 2 * xlarge / cur_h)) + 1
                cur_pos = cv2.resize(cur_pos, (scale * cur_pos.shape[1], scale * cur_pos.shape[0]))

    _, mag = highpass_and_imgback(cur_pos,xsmall, xlarge)
    mag = mag.reshape(-1)
    mag_norm = (mag-datamean)/datamax
    mag_projected = np.dot(mag_norm, uxx)

    mag_reconstructed = reconstruct_from_pca(mag_projected,ux,sx,vx,datamean, datamax,k)
    reconstructed_error = np.sum((mag_reconstructed-mag_norm)**2)

    if reconstructed_error > max_error or np.sqrt(np.sum(mag_projected[:k/2]**2)) < min_norm:
        return False
    else:
        #print '%f vs %f, %f vs %f'%(reconstructed_error,max_error, np.sqrt(np.sum(mag_projected[:k/2]**2)), min_norm)
        return True


pos_main_class = ['./annotation/Acanthamoeba_trophozoite_cyst.txt',
                  './annotation/Balantidium_cyst_trophozoite.txt',
                  './annotation/Cyclospora_oocyst_normal.txt',
                  './annotation/Cystoisospora_oocyst_single.txt',
                  './annotation/Giardia_trophozoite.txt',
                  './annotation/Iodamoeba_cyst.txt',
                  './annotation/Sarcocystis_oocyst_single.txt',
                  './annotation/Toxoplasma_cyst_single.txt'
                 ]

pos_files_group = [
             [#'./annotation_subclass/Sarcocystis_oocyst_single_stage2.txt',
             #'./annotation_subclass/Sarcocystis_oocyst_single_stage3.txt',
             './annotation_subclass/Sarcocystis_oocyst_stage2.txt',
             './annotation_subclass/Sarcocystis_oocyst_stage3.txt'],
             [#'./annotation_subclass/Cystoisospora_oocyst_single_stage1.txt',
             #'./annotation_subclass/Cystoisospora_oocyst_single_stage2.txt',
             #'./annotation_subclass/Cystoisospora_oocyst_single_stage3.txt',
             #'./annotation_subclass/Cystoisospora_oocyst_single_stage4.txt',
             './annotation_subclass/Cystoisospora_oocyst_stage1.txt',
             './annotation_subclass/Cystoisospora_oocyst_stage2.txt',
             './annotation_subclass/Cystoisospora_oocyst_stage3.txt',
             './annotation_subclass/Cystoisospora_oocyst_stage4.txt'],
             ['./annotation/Iodamoeba_cyst.txt'],
             ['./annotation_subclass/Cyclospora_oocyst_stage1.txt',
             './annotation_subclass/Cyclospora_oocyst_stage2.txt',
             './annotation_subclass/Cyclospora_oocyst_stage3.txt'],
             ['./annotation/Toxoplasma_cyst_single.txt'],
            ]
names = ['Sarcocystis','Cystoisospora','Iodamoeba','Cyclospora','Toxoplasma']
pos_files, group = creat_posfile_group(pos_files_group)
numclass = len(group)
num_label = len(pos_files)
numlabel = len(pos_files)


with open("myModel.pickle",'rb') as f:
    classifier,names,sz,ux,sx,vx,datamean, datamax, k,comp_parameters,root_filters,max_errors,min_norms,xlarge,xsmall = pickle.load(f)
uxx = ux[:,:k]
detection_dirs = ['./data/Sarcocystis_oocyst/',
                  './data/Iodamoeba_cyst/',
                  './data/Toxoplasma_cyst/',
                  './data/Cyclospora_oocyst/',
                  './data/Cystoisospora_oocyst/']

list_images = []
for detection_dir in detection_dirs:
    only_files = [detection_dir + f for f in listdir(detection_dir) if isfile(join(detection_dir,f))]
    list_images.extend(only_files)

colors = [(0,0,255),(0,255,0),(255,0,0),(255,0,255),(0,255,255),(255,255,0)]
scales = [1,2,3,4,5,6]
save_dir = './result/'
for file_dir_idx in range(len(list_images)):
    print '=========== %d / %d =============='%(file_dir_idx+1,len(list_images))
    file_dir = list_images[file_dir_idx]
    print file_dir

    tag = os.path.splitext(basename(file_dir))[0]
    image = cv2.imread(file_dir)

    hypotheses = []
    img = image.copy()

    for scale in scales:
        resized = cv2.resize(image,(0,0), fx=1./scale, fy=1./scale)
        img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        for filter_sz_idx in range(len(sz)):
            start_time = time.time()
            img = resized.copy()
            filter_sz = sz[filter_sz_idx]

            comp_ux,comp_sx,comp_vx,comp_datamean, comp_datamax, comp_k = comp_parameters[filter_sz_idx]
            comp_uxx = comp_ux[:,:comp_k]
            #print comp_k

            #for _i in range(0,img_gray.shape[0],int(filter_sz[0])/2 ):
            #    for _j in range(0,img_gray.shape[1],int(filter_sz[1])/2 ):
            for _i in range(0,img_gray.shape[0],int(filter_sz[0])/8):
                for _j in range(0,img_gray.shape[1], int(filter_sz[1])/8):
                    hypothesis = img_gray[_i:np.minimum(_i+int(filter_sz[0]), img_gray.shape[0]),
                                       _j:np.minimum(_j+int(filter_sz[1]), img_gray.shape[1])]
                    #if check_foreground(hypothesis, max_errors[filter_sz_idx],min_norms[filter_sz_idx],
                    #                    xsmall, xlarge,ux,sx,vx,datamean, datamax,k):
                    if check_foreground_hog(hypothesis, root_filters[filter_sz_idx],filter_sz) and check_foreground(hypothesis, max_errors[filter_sz_idx],min_norms[filter_sz_idx],
                                        xsmall, xlarge,ux,sx,vx,datamean, datamax,k):
                    #if check_foreground_hog(hypothesis, root_filters[filter_sz_idx],filter_sz):

                        #print hypothesis.shape
                        _, featureMagnitude = highpass_and_imgback(hypothesis,xsmall,xlarge)
                        featureMagnitude_norm = (featureMagnitude.reshape(-1)-datamean)/datamax
                        featureMagnitude_projected = np.dot(featureMagnitude_norm, uxx)
                        category = classifier.predict([featureMagnitude_projected])[0]
                        #==========================================================================
                        #featureMagnitude_norm = (featureMagnitude.reshape(-1)-comp_datamean)/comp_datamax

                        #featureMagnitude_projected = np.dot(featureMagnitude_norm, comp_uxx)

                        #category = comp_classifiers[filter_sz_idx].predict([featureMagnitude_projected])[0]
                        #==========================================================================
                        group_category = unify_label([category],group)[0]
                        #cv2.rectangle(img,(_j,_i),(_j+hypothesis.shape[1],_i+hypothesis.shape[0]),(0,255,0),3)
                        cv2.rectangle(img,(_j,_i),(_j+hypothesis.shape[1],_i+hypothesis.shape[0]),colors[group_category],3)

            cv2.imwrite(save_dir + tag + '_hog_duplicate_hardNegative_res_detected_scale_%d_filter_%d.png'%(scale,filter_sz_idx),img )
            print tag + '_hog_duplicate_res_detected_scale_%d_filter_%d.png'%(scale,filter_sz_idx)
            print 'Detection %s seconds'%(time.time()-start_time)
