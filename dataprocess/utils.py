import numpy as np


def findNonreflectiveSimilarity(src,dst):
    p=src
    A=np.zeros(40)
    for i in range(5):
        j =i*4
        A[j]=p[i]
        A[j+21]=p[i]*(-1)
        A[j+1]=A[j+20]=p[i+5]
        A[j+2] = A[j+23] = 1
        A[j+3] = A[j+22] = 0
    B=np.reshape(A,(10,4))
    # print(B)
    B=np.mat(B)
    A=B.I
    # print(A)
    A=np.array(A)
    M=np.dot(A,dst)

    M=np.array(M)
    m=np.zeros(6)

    m[0]=M[0]
    m[3]=M[1]*(-1)
    m[1]=M[1]
    m[2]=M[2]
    m[4]=M[0]
    m[5]=M[3]
    I=np.mat(np.reshape(m,(2,3)))
    
    # print(I)
    return I

import six
import cv2
from PIL import Image

def getlmdb_numpy(lmdb_txn,index):

    image_key = "image-%09d"%(index)
    label_key = "label-%09d"%(index)
    image_key = image_key.encode()

    image = lmdb_txn.get(image_key)
    label = lmdb_txn.get(label_key.encode())
    
    buf_str = six.BytesIO(image).getvalue()
    nparr = np.fromstring(buf_str, np.uint8)

    img_decode = cv2.imdecode(nparr, 1)
    # TC4: 测试读取的数据是否正确
    # cv2.imshow("test", img_decode)
    # cv2.waitKey(0)
    img = Image.fromarray(cv2.cvtColor(img_decode,cv2.COLOR_BGR2RGB))

    return img,label.decode()


def getlmdb_stream(lmdb_txn,index):

    image_key = "image-%09d"%(index)
    label_key = "label-%09d"%(index)
    image_key = image_key.encode()

    image = lmdb_txn.get(image_key)
    label = lmdb_txn.get(label_key.encode())
    
    buf = six.BytesIO()
    buf.write(image)
    buf.seek(0)
    # buf = image ?
    img = Image.open(buf).convert('RGB')
    return img,label.decode()

def getlmdb_numpy_image(lmdb_txn,index, key_str):

    image_key = key_str+"-%09d"%(index)
    image_key = image_key.encode()

    image = lmdb_txn.get(image_key)
    
    buf_str = six.BytesIO(image).getvalue()
    nparr = np.fromstring(buf_str, np.uint8)

    img_decode = cv2.imdecode(nparr, 1)
    # TC4: 测试读取的数据是否正确
    # cv2.imshow("test", img_decode)
    # cv2.waitKey(0)
    img = Image.fromarray(cv2.cvtColor(img_decode,cv2.COLOR_BGR2RGB))
    # img.show()
    # cv2.waitKey(0)
    return img


def getlmdb_numpy_label(lmdb_txn,index, key_str):

    label_key = key_str+"-%09d"%(index)
    #print(label_key)
    label = lmdb_txn.get(label_key.encode())

    return label.decode()


def getlmdb_caffe(lmdb_txn,index):

    sys.path.append('/home/ubuntu/data2/lk/amsoft_pytorch/mycaffe/python')
    import caffe
    lmdb_cursor=lmdb_txn.cursor()
    datum=caffe.proto.caffe_pb2.Datum()
    datum_index=0
    # IMP: 可以通过调查key的规律提高读数据的速度 #open
    for key,value in lmdb_cursor:
        if datum_index==index_id:
            datum.ParseFromString(value)
            # print(type(datum))# Datum类
            # print(datum)# 包含datum类的各个字段属性
            label=datum.label
            data=datum.data

            size=datum.width*datum.height
            pixles1=datum.data[0:size]
            pixles2=datum.data[size:2*size]
            pixles3=datum.data[2*size:3*size]
            #Extract images of different channel
            image1=Image.frombytes('L', (datum.width, datum.height), pixles1)
            image2=Image.frombytes('L', (datum.width, datum.height), pixles2)
            image3=Image.frombytes('L', (datum.width, datum.height), pixles3)
            #注意三通道的顺序，如果LMDB中图像是按照BGR存储的则需要按照：image3,image2,image1的顺序合并为RGB图像。PIL中图像是按照RGB的顺序存储的
            img=Image.merge("RGB",(image3,image2,image1))
        else:
            continue
    return img,label.decode()
