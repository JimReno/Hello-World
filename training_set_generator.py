import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy.io as sio
import glob,os
from PIL import Image

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

path = '/home/test/work/bank number classification'
#change the folder directory if needed
tfrecords_filename = path + '/tfrecords/train.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
width = 28
height = 28
#raw_training data directory
class_path = path+'/raw_data/train/train/'

#purify the image mode and size 
print "purifying the image"
for infile in os.listdir(class_path): 
    file, ext = os.path.splitext(infile)
    img_path=class_path+infile
    img = Image.open(img_path)
    img = img.convert("RGB")
    #print img.mode
    img=img.resize((width,height))
    #print img.size
    img.save(path + '/purified_data/train/' + file + '.jpg', 'JPEG')   
    #different filter can be chosen
    #img=img.resize((width,height),Image.ANTIALIAS)
    
print "Successfully resized all image to  %d * %d" %(width,height)
#generate tfrecords file

print "Generating training set..."
txtfile = path + '/purified_data/trainlist_jpg.txt'
fr = open(txtfile)
for i in fr.readlines():
    item = i.split()
    #print txtfile
    img = np.float64(misc.imread(path + '/purified_data/train/' + item[0] + ' ' + item[1]))
    #print path + '/raw_data/train/train/' + item[0] + ' ' + item[1]
    label = int(item[0])
    img_raw = img.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'name': _bytes_feature(item[0]+item[1]),
        'image_raw': _bytes_feature(img_raw),
        'label': _int64_feature(label)}))
    writer.write(example.SerializeToString())
writer.close()
fr.close()
print 'Successfully generated training set!'
