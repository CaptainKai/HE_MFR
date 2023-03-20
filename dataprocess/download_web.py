import  os
import urllib.request

import queue
import threading

import os,base64
import requests as req
from PIL import Image
from io import BytesIO

class myThread (threading.Thread):
    def __init__(self, q, qlock, llock, exitFlag=0):
        threading.Thread.__init__(self)
        self.q = q
        self.qlock = qlock
        self.exitFlag = exitFlag

        # self.recorder = Recorder()
    
    def run(self):
        self.process_data()

    def process_data(self):
        queueLock = self.qlock
        q = self.q

        while not self.exitFlag:
            queueLock.acquire()
            line = q.readline().strip()
            if line:
                index, id, url = line.split()
                print(index)
                pic_name = url.split("/")[-1].split("?")[0]
                pic_save_path = os.path.join(save_path, id, pic_name)
                if not os.path.exists(os.path.dirname(pic_save_path)):
                    os.makedirs(os.path.dirname(pic_save_path))
                queueLock.release()
                
                if os.path.exists(pic_save_path):
                    continue
                try:
                    # 法一：
                    # urllib.request.urlretrieve(url, filename=pic_save_path) # 不是并行,没有设置时间限制
                    # 法二： 不是并行，没有设置时间限制，报错信息更详细
                    response = req.get(url) # 将这个图片保存在内存
                    # 将这个图片从内存中打开，然后就可以用Image的方法进行操作了
                    # image = Image.open(BytesIO(response.content)) 
                    # 得到这个图片的base64编码
                    ls_f=base64.b64encode(BytesIO(response.content).read()).decode('utf-8')
                    imgdata=base64.b64decode(ls_f)
                    file=open(pic_save_path,'wb')
                    file.write(imgdata)
                    # # 关闭这个文件
                    # file.close()

                except Exception as e:
                    # logLock.acquire()
                    print(e)
                    # log_file.write(line+" "+str(e))
                    # logLock.release()
                
            else:
                queueLock.release()
                self.exit()

    def exit(self):
        self.exitFlag=1



save_path = "/home/ubuntu/data3/data/WebFace_260M_full"
url_list = "/home/ubuntu/data3/data/test.list"
# url_list = "/home/ubuntu/data3/data/WebFace_260M_URLs_All.lst"
log_path = "/home/ubuntu/data3/data/WebFace_260M_URLs_All.log"
url_list_file = open(url_list, "r")
# log_file = open(log_path, "a+")
# url_list_lines = url_list_file.readlines()
# line = url_list_file.readline().strip()
# while line:
#     if index<start_index:
#         line = url_list_file.readline().strip()
#         index += 1
#         continue
#     index, id, url = line.split()
#     pic_name = url.split("/")[-1]
#     dirname = id
#     pic_save_path = os.path.join(save_path, id, pic_name)
#     if not os.path.exists(os.path.dirname(pic_save_path)):
#         os.makedirs(os.path.dirname(pic_save_path))
#     下载并存储
#     try:
#         urllib.request.urlretrieve(url, filename=pic_save_path)
#         log_file.write(str(index)+" "+line)
#     except Exception as e:
#         log_file.write(str(index)+" "+line+" "+str(e))
queueLock = threading.Lock()
logLock = threading.Lock()
threads = []
thread_num = 12

# 创建新线程
for i in range(thread_num):
    thread = myThread(q=url_list_file, qlock=queueLock, llock=logLock)
    thread.start()
    threads.append(thread)

for t in threads: # 并没有结束各子线程运行
    t.join()

print("download done!")
