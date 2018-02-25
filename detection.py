
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
from os.path import basename
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


xsmall = 2
xlarge = 40

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
pos_files, group = creat_posfile_group(pos_files_group)
numclass = len(group)
num_label = len(pos_files)
numlabel = len(pos_files)
raw_poses = []
neg_images = []
for i in range(num_label):
    pfile = pos_files[i]
    raw_poses.append(get_pos(pfile))

for _i in pos_main_class:
    raw_neg_images = get_neg_images(_i)
    for neg_image in raw_neg_images:
        neg_images.append(cv2.cvtColor(neg_image, cv2.COLOR_BGR2GRAY))

#poses = []
#for _i in range(num_label):
#    poses.append(get_sample(raw_poses[_i],xsmall,xlarge))
numtrain = 0
ratio_test = 0.6
poses_train = []
poses_test = []
num_sample_train = []
for i in range(numlabel):
    poses = raw_poses[i]
    size = len(poses)
    train_size = np.minimum(int(np.ceil(ratio_test * size)), 3)
    if size == 2:
        train_size = 1
    numtrain += train_size
    num_sample_train.append(numtrain)

    idx = range(size)
    np.random.shuffle(idx)
    for t in idx[:train_size]:
        poses_train.append((poses[t],i))
    for t in idx[train_size:]:
        poses_test.append((poses[t],i))

samples_train = get_sample(poses_train,xsmall, xlarge)
samples_test = get_sample(poses_test,xsmall,xlarge)


# In[4]:


#num component
n = 10
INF = 10e8


pos = [p[2] for p in samples_train]
h = [p.shape[0]*1.0 for p in pos]
w = [p.shape[1]*1.0 for p in pos]
aspects = [_h/_w for _h, _w in zip(h,w)]
aspects = np.sort(aspects)

_interval = int(np.floor((len(aspects)-1)/(1.0*n)) + 1)

b = []
for _i in range(0,len(aspects), _interval):
    b.append(aspects[_i])
b.append(INF)

aspects = [_h/_w for _h, _w in zip(h,w)]

spos = []
for _i in range(len(b) - 1):
    spos.append([p for p,a in zip(samples_train,aspects) if a >= b[_i] and a < b[_i+1]])

print len(pos)
print len(spos[4])


# In[5]:


sbin = 1
sz = []
for _sp in spos:
    #pick mode of aspect ratios
    pos = [p[2] for p in _sp]
    h = [p.shape[0]*1.0 for p in pos]
    w = [p.shape[1]*1.0 for p in pos]
    xx = range(-200,200,2)
    xx = [x/100.0 for x in xx]
    _filter = np.exp( (-1.0)*np.array(range(-100,100,1))**2/400)
    aspects = np.histogram([np.log(_h/_w) for _h,_w in zip(h,w)], xx)

    aspects = np.convolve(aspects[0],_filter,mode='same')

    peak_idx = np.argmax(aspects)
    aspect = np.exp(xx[peak_idx])

    #pick 20 percentile area
    areas = np.sort([_h*_w for _h,_w in zip(h,w)])
    area = areas[int(np.floor(len(areas)*0.2))]

    #area = np.maximum(np.minimum(area,5000),3000)
    #print area

    #pick dimensions
    w = np.sqrt(area/aspect)
    h = w*aspect

    sz.append((np.round(h/sbin), np.round(w/sbin)))

print sz


# In[6]:


def warppos(fsize,sbin, pos):
    fh = fsize[0]*sbin*1.0
    fw = fsize[1]*sbin*1.0
    h = pos.shape[0]
    w = pos.shape[1]

    cropsize_x =  (fsize[1]+2)*sbin
    cropsize_y =  (fsize[0]+2)*sbin

    padx = sbin*w/fw
    pady = sbin*h/fh

    window = np.pad(pos, ((int(pady),int(pady)),(int(padx),int(padx))), 'edge')
    scaled = cv2.resize(window, (int(cropsize_y),int(cropsize_x)), interpolation = cv2.INTER_LINEAR)

    center_y = int(cropsize_y/2)
    center_x = int(cropsize_x/2)
    warped = scaled[center_y-int(fh)/2:center_y+int(fh)/2, center_x-int(fw)/2:center_x+int(fw)/2 ]

    return warped


# In[7]:


unique_sz = list(set(sz))
print unique_sz


# In[8]:


#check bounding box of root filters
image = cv2.imread('./data/Sarcocystis_oocyst/Sar_ooc_human_sto_x40_1_cut.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
filter_sz = unique_sz[0]
cc = 0
for filter_sz in unique_sz:
    img = image.copy()
    cc += 1
    for _i in range(0,image.shape[0], int(filter_sz[0])):
        for _j in range(0, image.shape[1], int(filter_sz[1])):
            cv2.rectangle(img,(_j,_i),(_j+int(filter_sz[1]),_i+int(filter_sz[0])),(0,255,0),3)

    #cv2.imwrite('./testcode/test%d.png'%(cc),img )
    #plt.imshow(image)
    #plt.show()


# In[9]:


pca_keep = 0.9
train_samples = np.array([x.reshape(-1) for x,l,p in samples_train])
train_samples_labels = np.array([l for x,l,p in samples_train])
train_samples_unified_labels = unify_label(train_samples_labels,group)

ux,sx,vx,datamean, datamax, k = customPCA(train_samples, pca_keep)

uxx = ux[:,:k]
X_train_norm = (train_samples - datamean) / datamax
X_train_projected = np.dot(X_train_norm, uxx)
#X_train_projected = np.dot(X_train_norm, uxx)/np.sqrt(sx[:k])


# In[10]:


def KNN(trn, trn_label,num_label,group):
    n_neighbor = 5
    #clf = neighbors.KNeighborsClassifier(n_neighbor, weights='distance')
    clf = neighbors.KNeighborsClassifier(n_neighbor, weights='distance',metric=mydistance)
    clf.fit(trn, trn_label)

    return clf


# In[11]:


#get classifier
classifier = KNN(X_train_projected, train_samples_labels,num_label,group)
print 'Done train classifiers'


# In[12]:


def reconstruct_from_pca(x,ux,sx,vx,datamean, datamax,k):
    N = ux.shape[1]
    padded = np.zeros((N))
    padded[:k] = x

    reconstructed = np.dot(ux,padded)
    return reconstructed

"""
reconstructed = []
for trn_sample in X_train_projected:
    reconstructed.append(reconstruct_from_pca(trn_sample,ux,sx,vx,datamean, datamax,k))

reconstructed = np.array(reconstructed)
max_error = np.max(np.sum((X_train_norm - reconstructed)**2,axis=1))
"""

max_errors = []
min_norms = []
for _sp in spos:
    sp_train_samples = np.array([x.reshape(-1) for x,l,p in _sp])
    sp_train_samples_norm = (sp_train_samples - datamean) / datamax
    sp_train_samples_projected = np.dot(sp_train_samples_norm, uxx)
    min_norms.append(np.min(np.sqrt(np.sum(sp_train_samples_projected[:,:k]**2, axis=1))))
    reconstructed = []
    for trn_sample in sp_train_samples_projected:
        reconstructed.append(reconstruct_from_pca(trn_sample,ux,sx,vx,datamean, datamax,k))

    reconstructed = np.array(reconstructed)
    max_error = np.max(np.sum((sp_train_samples_norm - reconstructed)**2,axis=1))
    max_errors.append(max_error)



# In[ ]:


def getNegatives(neg_images, fsize, numNeg):
    negs = []
    for _k in range(numNeg):
        _bg_idx = random.randint(0,len(neg_images)-1)
        _bg = neg_images[_bg_idx]
        _i = random.randint(0,_bg.shape[0] - fsize[0]-1)
        _j = random.randint(0,_bg.shape[1] - fsize[1]-1)
        negs.append(_bg[_i:int(_i+fsize[0]), _j:_j+int(fsize[1])])
    return negs

def get_one_HardNegative_by_rescale(fsize, pos, neg_image):
    neg = neg_image.copy()
    scale = random.randint(1,7)/10.0
    new_pos = cv2.resize(pos, (int(fsize[1]*scale),int(fsize[0]*scale)))
    _i = random.randint(0, int(fsize[0] - new_pos.shape[0] - 1))
    _j = random.randint(0, int(fsize[1] - new_pos.shape[1] - 1))
    neg[_i:_i+new_pos.shape[0],_j:_j+new_pos.shape[1]] = new_pos
    return neg

def get_one_HardNegative_by_translate(fsize,pos):
    new_pos = cv2.resize(pos, (int(fsize[1]),int(fsize[0])))
    _i = random.randint(int(0.2*new_pos.shape[0]), int(0.8*new_pos.shape[0]))
    _j = random.randint(int(0.2*new_pos.shape[1]), int(0.8*new_pos.shape[1]))
    res = new_pos.copy()
    res[:,res.shape[1]-_j:] = new_pos[:,:_j]
    res[:,:res.shape[1]-_j] = new_pos[:,_j:]
    new_pos = res.copy()
    res[res.shape[0]-_i:,:] = new_pos[:_i,:]
    res[:res.shape[0]-_i,:] = new_pos[_i:,:]
    return res

def get_HardNegatives(sz,spos, _idx, neg_images, numHardNeg):
    res_rescale = []
    res_translate = []
    fsize = sz[_idx]
    for _k in range(numHardNeg):
        neg_image = getNegatives(neg_images,fsize,1)[0]
        spos_idx = random.randint(0,len(spos)-1)
        pos_idx = random.randint(0, len(spos[spos_idx]) - 1)
        pos = spos[spos_idx][pos_idx][2]
        res_rescale.append(get_one_HardNegative_by_rescale(fsize, pos,neg_image))
        res_translate.append(get_one_HardNegative_by_translate(fsize,pos))
    return res_rescale,res_translate


def extract_pos(pos, fsize):
    num_rand = 4
    res = []
    cur_pos = pos
    cur_h, cur_w = cur_pos.shape
    if (cur_w < int(fsize[1])) or (cur_h < int(fsize[0])):
        scale = max(fsize[1] / cur_w,fsize[0] / cur_h)
        cur_pos = cv2.resize(cur_pos,(0,0),fx =scale, fy=scale)
        cur_h, cur_w = cur_pos.shape

    cx = cur_w/2
    cy = cur_h/2
    res.append(cur_pos[cy-int(fsize[0])/2:cy+int(fsize[0])/2,cx-int(fsize[1])/2:cx+int(fsize[1])/2])
    if cur_w > fsize[1] or cur_h > fsize[0]:
        for _i in range(num_rand):
            ncx = random.randint(cx-(cur_w - int(fsize[1]))/2, cx+(cur_w - int(fsize[1]))/2)
            ncy = random.randint(cy-(cur_h - int(fsize[0]))/2, cy+(cur_h - int(fsize[0]))/2)
            res.append(cur_pos[ncy-int(fsize[0])/2:ncy+int(fsize[0])/2,ncx-int(fsize[1])/2:ncx+int(fsize[1])/2])

    return res



root_filters = []
comp_classifiers = []
comp_parameters = []
for _idx in range(len(spos)):
    _sp = spos[_idx]
    _sz = sz[_idx]

    print _idx
    print 'Start computing HoG of positives'
    start_time = time.time()


    root_X = [feature.hog(cv2.resize(pos[2], (int(_sz[1]),int(_sz[0]))),
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          transform_sqrt=True,
                          visualise=True,
                          feature_vector=False)[0]
              for pos in _sp]
    print 'len root_X: %d'%(len(root_X))
    print 'Done HoG of positives %s seconds'%(time.time()-start_time)


    #==================================================================
    print 'Start computing data augmentation of HoG of positives'
    start_time = time.time()
    _nsp = []
    for p in _sp:
        _nsp.append((p[0],p[1],cv2.resize(p[2], (int(_sz[1]),int(_sz[0])))))
        #warped = warppos(_sz, sbin,p[2])
        ##warped = p[2]
        #extracted = extract_pos(warped,_sz)
        #for ext in extracted:
        #    _, ext_mag = highpass_and_imgback(ext,xsmall,xlarge)
        #    _nsp.append((ext_mag,p[1],ext))
    print '     Done extracting positives, Start computing HoG'
    print 'len root_X: %d'%(len(_nsp))
    #root_X = [feature.hog(pos[2],
    root_X = [feature.hog(cv2.resize(pos[2], (int(_sz[1]),int(_sz[0]))),
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          transform_sqrt=True,
                          visualise=True,
                          feature_vector=False)[0]
              for pos in _nsp]
    print 'Done data augmentation of HoG of positives %s seconds'%(time.time()-start_time)

    pca_keep = 0.9
    comp_train_samples = np.array([x.reshape(-1) for x,l,p in _nsp])
    comp_train_samples_labels = np.array([l for x,l,p in _nsp])
    comp_train_samples_unified_labels = unify_label(comp_train_samples_labels,group)

    comp_ux,comp_sx,comp_vx,comp_datamean, comp_datamax, comp_k = customPCA(comp_train_samples, pca_keep)
    comp_parameters.append((comp_ux,comp_sx,comp_vx,comp_datamean, comp_datamax, comp_k))
    comp_uxx = comp_ux[:,:comp_k]
    comp_X_train_norm = (comp_train_samples - comp_datamean) / comp_datamax
    comp_X_train_projected = np.dot(comp_X_train_norm, comp_uxx)
    #print comp_k
    #print comp_X_train_projected.shape
    #comp_classifier = KNN(comp_X_train_projected, comp_train_samples_labels,num_label,group)
    #comp_classifiers.append(comp_classifier)
    #==================================================================


    #for epoch in range(1000):
    X = []
    Y = []
    for x in root_X:
        #print x.shape
        X.append(x.reshape(-1))
        Y.append(1)

    #negs = getNegatives(neg_images,_sz,len(_sp))
    #=================================================================================================
    print 'Start getting Negatives'
    start_time = time.time()
    negs = []
    for neg in getNegatives(neg_images,_sz,6*len(_sp)):
        negs.append(neg)
    print 'Done Negative %s seconds'%(time.time()-start_time)


    print 'Start getting Hard Negative'
    start_time = time.time()
    res_rescales,res_translates = get_HardNegatives(sz,spos, _idx, neg_images, 3*len(_sp))
    for neg_rescale, neg_translate in zip(res_rescales,res_translates):
        negs.append(neg_rescale)
        negs.append(neg_translate)
    print 'Done Hard Negative %s seconds'%(time.time()-start_time)
    #==================================================================================================
    print 'Start getting HoG of Negatives'
    print 'len negs: %d'%(len(negs))
    start_time = time.time()
    for x in negs:
        #hogFeature = feature.hog(x,
        #print len(x)
        hogFeature = feature.hog(cv2.resize(x, (int(_sz[1]),int(_sz[0]))),
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      transform_sqrt=True,
                      visualise=True,
                      feature_vector=False)[0]
        #print hogFeature.shape
        X.append(hogFeature.reshape(-1))
        Y.append(0)

    print 'Done HoG of Negatives %s seconds'%(time.time()-start_time)
    print 'Start getting HoG classifier'
    start_time = time.time()

    #clf = svm.LinearSVC(max_iter = -1, class_weight={1:10})
    #clf = svm.LinearSVC(max_iter = 200)
    #clf = svm.LinearSVC(max_iter = 200,class_weight={1:2})
    clf = svm.LinearSVC(max_iter = 200,class_weight={1:len(negs)/len(root_X)})
    clf.fit(np.array(X) ,np.array(Y))
    root_filters.append(clf)

    print 'Done HoG classifier %s seconds'%(time.time()-start_time)


# In[ ]:


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

#image = cv2.imread('./data/Sarcocystis_oocyst/Sar_ooc_human_sto_x40_1_cut.jpg')
#file_dir = './data/Cyclospora_oocyst/Cca_ooc_human_sto_x20_124_split_2.jpg'
#file_dir = './data/Toxoplasma_cyst/Tgo_cys_hts_x40_2_cut.jpg'
#file_dir = './data/Iodamoeba_cyst/Ibu_cys_human_sto_x40_003.jpg'
#file_dir = './data/Iodamoeba_cyst/Ibu_cys_human_sto_x40_010.jpg'
file_dir = './data/Iodamoeba_cyst/Ibu_cys_human_sto_x40_014.jpg'
file_dir = './data/Iodamoeba_cyst/Ibu_cys_human_sto_x40_015.jpg'
#file_dir = './data/Sarcocystis_oocyst/Sar_ooc_human_sto_x40_1_cut.jpg'
tag = os.path.splitext(basename(file_dir))[0]
image = cv2.imread(file_dir)

hypotheses = []
img = image.copy()
colors = [(0,0,255),(0,255,0),(255,0,0),(255,0,255),(0,255,255),(255,255,0)]
scales = [1,2,3,4,5,6]
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

        cv2.imwrite('./testcode/' + tag + '_hog_duplicate_hardNegative_res_detected_scale_%d_filter_%d.png'%(scale,filter_sz_idx),img )
        print tag + '_hog_duplicate_res_detected_scale_%d_filter_%d.png'%(scale,filter_sz_idx)
        print 'Detection %s seconds'%(time.time()-start_time)
