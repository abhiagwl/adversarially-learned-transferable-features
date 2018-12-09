import pickle as pkl
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

root = "../"
data_dir = os.path.join(root,"datasets/mnistm/")

mnistm = pkl.load(open(data_dir+"mnistm_data.pkl", 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']

train_labels = mnist.train.labels
train_labels = np.argmax(train_labels,1)
test_labels = np.argmax(mnist.test.labels,1)
np.savetxt("./datasets/mnistm/train/train_labels.txt",train_labels)
np.savetxt("./datasets/mnistm/test/test_labels.txt",test_labels)

def save_mnistm_png(data,train = True,data_dir =""):
    if train == True:
        for i in xrange(len(data)):
            Image.fromarray(data[i]).save(data_dir+"train/"+str(i)+".png")
    else:
        for i in xrange(len(data)):
            Image.fromarray(data[i]).save(data_dir+"test/"+str(i)+".png")
            
save_mnistm_png(mnistm_train,True,data_dir)
save_mnistm_png(mnistm_test,False,data_dir)