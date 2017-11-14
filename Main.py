#encoding: utf-8
import sys
sys.path.append("/Users/tianpeiqi/caffe")
sys.path.append("/Users/tianpeiqi/caffe/python")
import caffe
from caffe import Classifier
import numpy as np
import os
#import cv2
import cv2 as cv
from skimage import transform as tf
from skimage import io

from PIL import Image, ImageDraw
import threading
from time import ctime,sleep
import time
import sklearn

import sklearn.metrics.pairwise as pw

from imutils.video import VideoStream
from imutils.video import FPS
import imutils



#保存人脸的位置

#我把GPU加速注释掉了,所以没有GPU加速,速度有点慢,你要在学校有条件找个有GeForce显卡的电脑
#caffe.set_mode_gpu()

#加载caffe模型
global net
net = Classifier('/Users/tianpeiqi/VGG_Face_Caffe_Model/deploy.prototxt', '/Users/tianpeiqi/VGG_Face_Caffe_Model/vgg_face.caffemodel')

#用来显示当前图片
#Opencv中人脸检测的一个级联分类器
cascade = cv.CascadeClassifier("/Users/tianpeiqi/VGG_Face_Caffe_Model/haarcascade_frontalface_alt.xml")
#获取视频流的接口，0表示摄像头的id号，当只连接一个摄像头时默认为0
source = "rtsp://admin:12345abcde@10.10.20.37"
#camera = cv.VideoCapture(source)


def detectFaces(img):

    face_cascade = cascade
    if img.ndim == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result

def drawFaces(img):

    faces = detectFaces(img)
    if faces:
        for (x1,y1,x2,y2) in faces:
            cv.rectangle(img,(x1,y1),(x2,y2),(255,105,65),3)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, 'FaceDetection', (x1,y1-5), font, 1, (255,0,255), 2)
    return img


#用来注册一个用户,要求图片大小为224*224
def register(path, reg_id, img):
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (224,224), interpolation=cv.INTER_AREA)

    cv.imwrite(path, img)

#用来识别一个用户
def recog(md,img):

    #global face_rect
    src_path='./regist_pic/'+str(md)
    #while True:
     #   rects=face_rect
      #  if rects:
       #     #img保存用来验证的人脸
        #    if rects[0][2]<rects[0][3]:
         #       cv.SetImageROI(img,(rects[0][0]+10, rects[0][1]+10,rects[0][2]-100,rects[0][2]-100))
          #  else:
           #     cv.SetImageROI(img,(rects[0][0]+10, rects[0][1]+10,rects[0][3]-100,rects[0][3]-100))
            ##将img暂时保存起来
            #dst=cv.CreateImage((224,224), 8, 3)
            #cv.resize(img,dst,cv.INTER_LINEAR)
            #cv.SaveImage('./temp.bmp',dst)
            #取出5张注册的人脸,分别与带验证的人脸进行匹配,可以得到五个相似度,保存到scores中
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imwrite('./temp.png', img)
    scores=[]
    for i in range(5):
        res=compar_pic('./temp.png',src_path+'/'+str(i)+'.png')
        scores.append(res)
        print res
    #求scores的均值
    result=avg(scores)
    print 'avg is :',avg(scores)
    return result


def avg(scores):

    max=scores[0]
    min=scores[0]
    res=0.0
    for i in scores:
        res=res+i
        if min>i:
            min=i
        if max<i:
            max=i
    return (res-min-max)/3


def compar_pic(path1,path2):

    global net
    #加载验证图片
    X=read_image(path1)
    test_num=np.shape(X)[0]
    #X  作为 模型的输入
    out = net.forward_all(data=X)
    #fc7是模型的输出,也就是特征值
    feature1 = np.float64(out['fc7'])
    feature1 = np.reshape(feature1,(test_num,4096))
    #np.savetxt('feature1.txt', feature1, delimiter=',')

    #加载注册图片
    X=read_image(path2)
    #X  作为 模型的输入
    out = net.forward_all(data=X)
    #fc7是模型的输出,也就是特征值
    feature2 = np.float64(out['fc7'])
    feature2 = np.reshape(feature2,(test_num,4096))
    #np.savetxt('feature2.txt', feature2, delimiter=',')
    #求两个特征向量的cos值,并作为是否相似的依据
    predicts=pw.cosine_similarity(feature1, feature2)
    return  predicts



def read_image(filelist):

    averageImg = [129.1863,104.7624,93.5940]
    X = np.empty([1,3,224,224])
    word = filelist.split('\n')
    filename = word[0]
    im1 = io.imread(filename ,as_grey=False)
    image = tf.resize(im1,(224, 224))*255
    #print image
    X[0,0,:,:] = image[:,:,0]-averageImg[0]
    X[0,1,:,:] = image[:,:,1]-averageImg[1]
    X[0,2,:,:] = image[:,:,2]-averageImg[2]
    #print X
    return X



def show_img():

    while (1):
        img = camera.read()
        img = drawFaces(img)
        cv.imshow('DeepFace', img)
        c = cv.waitKey(1)
        if c&0xFF == ord('q'):
            break



if __name__ == '__main__':

    #人脸检测模块
    camera = VideoStream(source).start()
    img = camera.read()
    print '人脸识别模块开启，按Q键退出进入下一步...'
    show_img()
    #camera.release()
    cv.destroyWindow('DeepFace')

    while True:
        pattern=raw_input('注册输入1\n识别输入2\n请选择程序模式:')
        if pattern=='1':
            tag=0
            reg_id=raw_input('请输入注册id:')
            reg_path='./regist_pic'
            #判断用户是否已经注册
            dir_rec=os.listdir(reg_path)
            for subdir in dir_rec:
                if(subdir==reg_id):#说明该用户已经注册
                    print '该用户已经注册!!!!!!\n'
                    tag=1
            #该用户未注册
            if tag==0:
                #生成该用户的文件夹和注册图片
                #camera = cv.VideoCapture(0)
                time.sleep(2)
                os.mkdir(reg_path+'/'+reg_id)
                num=-2
                #注册五张人脸
                while num<4:
                    if True:
                        num=num+1
                        if num>=0:
                            register_path=reg_path+'/'+str(reg_id)+'/'+str(num)+'.png'
                            img = camera.read()    #注意read()函数返回值为二元组
                            register(register_path, reg_id, img)
                            print 'now is '+str(num)+'........\n'
                            time.sleep(2)


        if pattern=='2':
            #md保存验证的id
            md=raw_input('请输入识别id:')
            #判断该用户是否存在
            tag=0
            dir_rec=os.listdir('./regist_pic')
            for subdir in dir_rec:
                if(subdir==md):#说明该用户存在
                    tag=1
            if tag==1:
                print '请注视摄像头'

                #阈值,大于这个值说明两个类的距离较远,不是一类
                thershold=0.85
                #把捕捉到的图片与注册的图片比较
                time.sleep(5)
                result=recog(md, img)
                if result>=thershold:
                    print result
                    print '验证成功!!!!\n\n'
                else:
                    print '验证失败,不是本人!!!!\n\n'
            else:
                print '该用户不存在!!!!\n\n'



