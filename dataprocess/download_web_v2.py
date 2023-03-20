# -*-coding: utf-8 -*-
 
import time
from multiprocessing.pool import ThreadPool
import requests
import os
import PIL.Image as Image
from io import BytesIO
import cv2
import numpy as np
 
from urllib.parse import urlparse, parse_qs

def download_image(line, our_dir):
    '''
    根据url下载图片
    :param url:
    :return: 返回保存的图片途径
    '''
    index, id, url = line.split()
    print(index)
    # pic_name = url.split("/")[-1].split("?")[0] # v1 解析版本
    a = urlparse(url)
    
    file_path = a.path
    try: # 获取真正的url路径
        true_url = parse_qs(urlparse(url).query)["url"][0]
        # print(true_url)
        url = true_url
        a = urlparse(url)
        file_path = a.path
    except:
        pass
    _, file_suffix = os.path.splitext(file_path)
    
    try: # 获取图片后缀
        suffix = parse_qs(urlparse(url).query)["format"][0]
        # print(suffix)
        if file_suffix and file_suffix!=suffix:
            pass
        else:
            file_path = file_path+"."+suffix
    except Exception as e:
        pass
    
    pic_name = os.path.basename(file_path)# 消除"/"
    
    pic_save_path = os.path.join(our_dir, id, pic_name)
    if os.path.exists(pic_save_path):
        return pic_save_path
    if not os.path.exists(os.path.dirname(pic_save_path)):
        os.makedirs(os.path.dirname(pic_save_path))
    
    try:
        res = requests.get(url)
        if res.status_code == 200:
            # print("download image successfully:{}".format(url))
            # filename = os.path.join(our_dir, basename)
            with open(pic_save_path, "wb") as f:
                content = res.content
                # 使用Image解码为图片
                # image = Image.open(BytesIO(content))
                # image.show()
                # 使用opencv解码为图片
                content = np.asarray(bytearray(content), dtype="uint8")
                # image = cv2.imdecode(content, cv2.IMREAD_COLOR)
                # cv2.imshow("Image", image)
                # cv2.waitKey(1000)
                f.write(content)
                # time.sleep(2)
            return pic_save_path
    except Exception as e:
        print(e)
        return None
    print("download image failed:{}".format(url))
    return None
 
 
def download_image_thread(url_list_file, our_dir, num_processes, Async=True):
    '''
    多线程下载图片
    :param url_list: image url list
    :param our_dir:  保存图片的路径
    :param num_processes: 开启线程个数
    :param remove_bad: 是否去除下载失败的数据
    :param Async:是否异步
    :return: 返回图片的存储地址列表
    '''
    # 开启多线程
    pool = ThreadPool(processes=num_processes)
    line = url_list_file.readline().strip()
    while line:
        if Async:
            out = pool.apply_async(func=download_image, args=(line, our_dir))  # 异步
        else:
            out = pool.apply(func=download_image, args=(line, our_dir))  # 同步
        line = url_list_file.readline().strip()
 
    pool.close()
    pool.join()

if __name__ == "__main__":
    save_path = "/home/ubuntu/data3/data/WebFace_260M_full"
    url_list = "/home/ubuntu/data3/data/test.list"
    # url_list = "/home/ubuntu/data3/data/WebFace_260M_URLs_All.lst"
    # log_path = "/home/ubuntu/data3/data/WebFace_260M_URLs_All.log"
    url_list_file = open(url_list, "r")
    startTime = time.time()
    download_image_thread(url_list_file, our_dir=save_path, num_processes=12, Async=True)
    endTime = time.time()
    consumeTime = endTime - startTime
    print("程序运行时间：" + str(consumeTime) + " 秒")
    # print(image_list)