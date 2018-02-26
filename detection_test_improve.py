
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

def check_foreground_hog2(hypothesis_HoG, clf):
    predict = clf.predict([hypothesis_HoG.reshape(-1)])[0]
    if predict == 1:
        #print 'hello'
        return True
    else:
        return False

def check_foreground_hog3(hypothesis_HoG, clf, foregroundThresh):
    predict = clf.predict_proba([hypothesis_HoG.reshape(-1)])[0][1]
    if predict >= foregroundThresh:
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

    if reconstructed_error > max_error or np.sqrt(np.sum(mag_projected[:k]**2)) < min_norm:
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
             ['./annotation_subclass/Sarcocystis_oocyst_single_stage2.txt',
             './annotation_subclass/Sarcocystis_oocyst_single_stage3.txt',
             './annotation_subclass/Sarcocystis_oocyst_stage2.txt',
             #'./annotation_subclass/Sarcocystis_oocyst_stage3.txt'
             ],
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

print 'Loading model'
start_time = time.time()
with open("myModel_improved.pickle",'rb') as f:
    classifier,names,sz,ux,sx,vx,datamean, datamax, k,comp_parameters,root_filters,max_errors,min_norms,max_scales,min_scales,xlarge,xsmall,HoG_pixels_per_cell,HoG_cells_per_block = pickle.load(f)
uxx = ux[:,:k]
print 'Done loading model %s s'%(time.time() - start_time)
detection_dirs = ['./data/Sarcocystis_oocyst/',
                  './data/Iodamoeba_cyst/',
                  './data/Toxoplasma_cyst/',
                  './data/Cyclospora_oocyst/',
                  './data/Cystoisospora_oocyst/']




def nms_multiclass(hypotheses, overlap, numlabel):
    x1 = np.array([x for x,y,w,h,c,p in hypotheses])
    y1 = np.array([y for x,y,w,h,c,p in hypotheses])
    x2 = np.array([x+w for x,y,w,h,c,p in hypotheses])
    y2 = np.array([y+h for x,y,w,h,c,p in hypotheses])
    _c = np.array([c for x,y,w,h,c,p in hypotheses])
    area = (x2-x1+1)*(y2-y1+1) * 1.0
    #probs = np.zeros((len(hypotheses),num_label))
    probs = np.array([x for x,y,w,h,c,p in hypotheses])
    """
    for _i in range(len(hypotheses)):
        probs[_i][_c[_i]] = 1
    for _i in range(0,len(hypotheses)-1):
        for _j in range(_i+1,len(hypotheses)):
            xx1 = max(x1[_i],x1[_j])
            yy1 = max(y1[_i],y1[_j])
            xx2 = min(x2[_i],x2[_j])
            yy2 = min(y2[_i],y2[_j])
            w = xx2 - xx1 + 1
            h = yy2 - yy1 + 1
            if w > 0 and h > 0:
                o1 = w*h / area[_i]
                o2 = w*h / area[_j]
                if o1 >= overlap and o2 >= overlap:
                #o = w*h/(area[_i] + area[_j] - w*h*1.0)
                #if o > overlap:
                    probs[_i][_c[_j]] += 1
                    probs[_j][_c[_i]] += 1


    s = [probs[_i][_c[_i]]/np.sum(probs[_i]) for _i in range(len(hypotheses))]
    """
    s = probs
    I = np.argsort(s).tolist()
    pick = []
    while not I == []:
        _last = len(I) - 1
        _i = I[_last]
        pick.append(_i)
        suppress = [_last]
        for _pos in range(_last):
            _j = I[_pos]
            xx1 = max(x1[_i],x1[_j])
            yy1 = max(y1[_i],y1[_j])
            xx2 = min(x2[_i],x2[_j])
            yy2 = min(y2[_i],y2[_j])
            w = xx2 - xx1 + 1
            h = yy2 - yy1 + 1
            if w > 0 and h > 0:
                o = w*h / area[_j]
                if o >= overlap:
                    suppress.append(_pos)
        for _index in sorted(suppress, reverse=True):
            I.remove(I[_index])
    return [(hypotheses[p],s[p]) for p in pick]

def nms_multiclass_faster(hypotheses, overlap, numlabel):
    x1 = np.array([x for x,y,w,h,c,p in hypotheses])
    y1 = np.array([y for x,y,w,h,c,p in hypotheses])
    x2 = np.array([x+w for x,y,w,h,c,p in hypotheses])
    y2 = np.array([y+h for x,y,w,h,c,p in hypotheses])
    _c = np.array([c for x,y,w,h,c,p in hypotheses])
    area = (x2-x1+1)*(y2-y1+1) * 1.0
    #probs = np.zeros((len(hypotheses),num_label))
    probs = np.array([x for x,y,w,h,c,p in hypotheses])
    s = probs
    I = np.argsort(s)
    pick = []
    while len(I) > 0:
        _last = len(I) - 1
        _i = I[_last]
        pick.append(_i)
        suppress = [_last]
        xx1 = np.maximum(x1[_i],x1[I[:_last]])
        yy1 = np.maximum(y1[_i],y1[I[:_last]])
        xx2 = np.minimum(x2[_i],x2[I[:_last]])
        yy2 = np.minimum(y2[_i],y2[I[:_last]])

        w = np.maximum(0,xx2 - xx1 + 1)
        h = np.maximum(0,yy2 - yy1 + 1)
        o = (w*h) / area[I[:_last]]
        I = np.delete(I, np.concatenate(([_last],np.where(o>overlap)[0])))
    return [(hypotheses[p],s[p]) for p in pick]

def cumsum(a):
    return [np.sum(a[:_i+1]) for _i in range(len(a))]

def load_groundtruth(fi):
	res = []
	with open(fi) as f:
		lines = f.readlines()
		for line in lines:
			content = line.split()
			file_dir = content[0].replace('\\','/')
			num_object = content[1]
			file_bb = []
			for _i in range(int(num_object)):
				x = int(content[2+4*_i])
				y = int(content[2+4*_i + 1])
				w = int(content[2+4*_i + 2])
				h = int(content[2+4*_i + 3])
				file_bb.append((x,y,w,h))
			res.append((file_dir,file_bb))
	return res


list_images = []
for detection_dir in detection_dirs:
    only_files = [detection_dir + f for f in listdir(detection_dir) if isfile(join(detection_dir,f))]
    list_images.extend(only_files)

overlap = 0.64
overlap_nms = overlap
colors = [(0,0,255),(0,255,0),(255,0,0),(255,0,255),(0,255,255),(255,255,0)]
#scales = range(1,50)
scales = [1,1.5,1.75,2,2.5,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,24,32,48]
#scales = 2**(1/10)**np.array([1,2,3,4,5,6,7,8,9,10])
save_dir = './result/improve/'
detections = []
names = ['Sarcocystis','Cystoisospora','Iodamoeba','Cyclospora','Toxoplasma']
annotation_test_files = ['./annotation/' + name + '.txt' for name in names]
mAP = []
#========== for test code only ========================================
#list_images = ['./data/Iodamoeba_cyst/Ibu_cys_human_sto_x40_003.jpg',
#               './data/Iodamoeba_cyst/Ibu_cys_human_sto_x40_010.jpg']
#======================================================================
for file_dir_idx in range(len(list_images)):
    print '=========== %d / %d =============='%(file_dir_idx+1,len(list_images))
    file_dir = list_images[file_dir_idx]
    print file_dir

    tag = os.path.splitext(basename(file_dir))[0]
    image = cv2.imread(file_dir)

    hypotheses = []
    img = image.copy()

    start_detection = time.time()
    for scale in scales:
        resized = cv2.resize(image,(0,0), fx=1./scale, fy=1./scale)
        img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        try:
            hog_feature_map = feature.hog(img_gray,
                                  orientations=9,
                                  pixels_per_cell= HoG_pixels_per_cell,
                                  cells_per_block= HoG_cells_per_block,
                                  transform_sqrt=True,
                                  visualise=True,
                                  feature_vector=False)[0]
        except:
            print 'Cannot compute HoG feature map'
            continue
        for filter_sz_idx in range(len(sz)):
            #if max_scales[filter_sz_idx] < scale:
            #    continue
            start_time = time.time()
            #img = resized.copy()
            filter_sz = sz[filter_sz_idx]
            _sz_HoG = (filter_sz[0]/HoG_pixels_per_cell[1] - HoG_cells_per_block[1]+1, filter_sz[1]/HoG_pixels_per_cell[0] - HoG_cells_per_block[0]+1)

            comp_ux,comp_sx,comp_vx,comp_datamean, comp_datamax, comp_k = comp_parameters[filter_sz_idx]
            comp_uxx = comp_ux[:,:comp_k]
            #print comp_k

            #for _i in range(0,img_gray.shape[0],int(filter_sz[0])/2 ):
            #    for _j in range(0,img_gray.shape[1],int(filter_sz[1])/2 ):
            #for _i in range(0,img_gray.shape[0],int(filter_sz[0])/8):
            #    for _j in range(0,img_gray.shape[1], int(filter_sz[1])/8):
            for _i in range(hog_feature_map.shape[0] - int(_sz_HoG[0])):
                for _j in range(hog_feature_map.shape[1]- int(_sz_HoG[1])):

                    hypothesis_HoG = hog_feature_map[_i:int(_i+_sz_HoG[0]), _j:int(_j+_sz_HoG[1])]
                    #print hypothesis_HoG.shape

                    hypothesis = img_gray[int(_i*HoG_pixels_per_cell[1]):int(_i*HoG_pixels_per_cell[1]+ filter_sz[0]), int(_j*HoG_pixels_per_cell[0]):int(_j*HoG_pixels_per_cell[0]+ filter_sz[1])]
                    #if check_foreground(hypothesis, max_errors[filter_sz_idx],min_norms[filter_sz_idx],
                    #                    xsmall, xlarge,ux,sx,vx,datamean, datamax,k):
                    if check_foreground_hog2(hypothesis_HoG, root_filters[filter_sz_idx]):
                    #if check_foreground_hog2(hypothesis_HoG, root_filters[filter_sz_idx]) and check_foreground(hypothesis, max_errors[filter_sz_idx],min_norms[filter_sz_idx],
                    #                    xsmall, xlarge,ux,sx,vx,datamean, datamax,k):
                    #if check_foreground_hog3(hypothesis_HoG, root_filters[filter_sz_idx],foregroundThresh=0.7) and check_foreground(hypothesis, max_errors[filter_sz_idx],min_norms[filter_sz_idx],
                    #                    xsmall, xlarge,ux,sx,vx,datamean, datamax,k):
                    #if check_foreground_hog(hypothesis, root_filters[filter_sz_idx],filter_sz):

                        #print hypothesis.shape
                        _, featureMagnitude = highpass_and_imgback(hypothesis,xsmall,xlarge)
                        featureMagnitude_norm = (featureMagnitude.reshape(-1)-datamean)/datamax
                        featureMagnitude_projected = np.dot(featureMagnitude_norm, uxx)
                        category = classifier.predict([featureMagnitude_projected])[0]
                        #prob = classifier.predict_proba([featureMagnitude_projected])[0]
                        #print 'prob'
                        #print prob
                        prob = classifier.predict_proba([featureMagnitude_projected])[0][category]
                        #prob = classifier.predict_proba([featureMagnitude_projected])[0][category] * root_filters[filter_sz_idx].predict_proba([hypothesis_HoG.reshape(-1)])[0][1]
                        print 'prob'
                        print prob
                        #==========================================================================
                        #featureMagnitude_norm = (featureMagnitude.reshape(-1)-comp_datamean)/comp_datamax

                        #featureMagnitude_projected = np.dot(featureMagnitude_norm, comp_uxx)

                        #category = comp_classifiers[filter_sz_idx].predict([featureMagnitude_projected])[0]
                        #==========================================================================
                        group_category = unify_label([category],group)[0]
                        #cv2.rectangle(img,(_j,_i),(_j+hypothesis.shape[1],_i+hypothesis.shape[0]),(0,255,0),3)
                        #cv2.rectangle(img,(int(_j*scale),int(_i*scale)),(int((_j+hypothesis.shape[1])*scale),int(scale*(_i+hypothesis.shape[0]))),colors[group_category],3)
                        #hypotheses.append((_j*scale*HoG_pixels_per_cell[0],_i*scale*HoG_pixels_per_cell[1],hypothesis.shape[1]*scale, hypothesis.shape[0]*scale,group_category))
                        if prob > 0.8:
                            hypotheses.append((_j*scale*HoG_pixels_per_cell[0],_i*scale*HoG_pixels_per_cell[1],hypothesis.shape[1]*scale, hypothesis.shape[0]*scale,group_category,prob))
            print tag + '_hog_duplicate_res_detected_scale_%d_filter_%d.png'%(scale,filter_sz_idx)
            print 'Detection %s seconds'%(time.time()-start_time)
    temp_img = img.copy()
    for _hypo in hypotheses:
        x,y,w,h,c,prob = _hypo
        cv2.rectangle(temp_img,(int(x),int(y)),(int((x+w)),int(y+h)),colors[c],3)
    cv2.imwrite(save_dir + tag + '_detected_improved_before_nms.png',temp_img )
    #detection = nms_multiclass(hypotheses,overlap_nms, num_label)
    detection = nms_multiclass_faster(hypotheses,overlap_nms, num_label)
    print 'Detection %s seconds'%(time.time()-start_detection)
    for _hypo,_s in detection:
        x,y,w,h,c,prob = _hypo
        cv2.rectangle(img,(int(x),int(y)),(int((x+w)),int(y+h)),colors[c],3)
    cv2.imwrite(save_dir + tag + '_detected_improved.png',img )
    detections.append((file_dir,detection))

#print detections
with open(save_dir+'result_improved.pickle','wb') as f:
	pickle.dump(detections,f)

for _c in range(len(names)):
	gt = load_groundtruth(annotation_test_files[_c])
	gt_det = [np.zeros(len(gt[_idx_k][1])) for _idx_k in range(len(gt))]
	BB = []
	in_files = []
	for _bb in detections:
		file_dir = _bb[0]
		for _b in _bb[1]:
			if _b[0][4] == _c:
				BB.append(_b)
				in_files.append(file_dir)
	idxs = np.argsort([conf for bbox,conf in BB])
	#BBB = [BB[_idx][1] for _idx in idxs]
	BBB = [BB[_idx][0] for _idx in idxs]
	in_files = [in_files[_idx] for _idx in idxs]
	tp = np.zeros(len(BBB))
	fp = np.zeros(len(BBB))
	for _idx in range(len(BBB)):
		_idx_gt = -1
		_bb = BBB[_idx]
		for _i in range(len(gt)):
			if gt[_i][0] == in_files[_idx]:
				_idx_gt = _i
		if _idx_gt == -1:
			continue
		if gt[_idx_gt][1] == []:
			fp[_idx] = 1
		else:
			jmax = -1
			ovmax = 0
			for _j in range(len(gt[_idx_gt][1])):
				bbgt = gt[_idx_gt][1][_j]
				#print bbgt
				#print _bb
				bi = (max(_bb[0],bbgt[0]),max(_bb[1],bbgt[1]),
							min(_bb[0]+_bb[2],bbgt[0]+bbgt[2]),min(_bb[1]+_bb[3],bbgt[1]+bbgt[3]))
				iw = bi[2] - bi[0] + 1
				ih = bi[3] - bi[1] + 1
				if iw > 0 and ih > 0:
					#ua = (_bb[2]-_bb[0] + 1)*(_bb[3]-_bb[1] + 1) +(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1) - iw*ih
					ua = (_bb[2])*(_bb[3]) +(bbgt[2])*(bbgt[3]) - iw*ih*1.0
					ov = iw*ih/ua
					ov = iw*ih/ua
					if ov > ovmax:
						ovmax = ov
						jmax = _j
			if ovmax > overlap:
				if not gt_det[_idx_gt][jmax]:
					tp[_idx] = 1
					gt_det[_idx_gt][jmax] = 1
				else:
					fp[_idx] = 1
			else:
				fp[_idx] = 1
	fp = np.array(cumsum(fp))
	tp = np.array(cumsum(tp))
	npos = np.sum([len(_gt[1]) for _gt in gt])
	rec = tp*1.0/npos
	prec = tp*1.0/ (tp+fp + 10e-8)
	#print prec
	#print rec
	ap = 0
	for t in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
		p = [prec[_i] for _i in range(len(prec)) if rec[_i] > t]
		if p == []:
			p = 0
		else:
			p = max(p)
		ap = ap + p/11
	#mAP.append(ap)
	if not len(prec) == 0:
		mAP.append((ap,prec[-1],rec[-1]))
	else:
		mAP.append((ap,0,0))
print mAP
with open(save_dir+'mAP_improved.pickle','wb') as f:
	pickle.dump(mAP,f)
