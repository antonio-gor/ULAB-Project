################
## Packages to be used
####

from tensorflow import keras
from random import shuffle
import tensorflow as tf
import numpy as np
import glob
import time
import sys
import os

tf.enable_eager_execution()

################
## Methods to be used for tfrecords creation
####

## Wrappers for the feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    

def create_data_record(out_filename, data, labels, data_type='Train'):
    '''
    The create_data_record function takes in data and labels,
        then converts them to tfrecords.
    
    Args: 
        out_filename: str; location and ouput file name.
            Must end with the .tfrecords file type.
            Ex. out_filename = 'tfrecords/train.tfrecords'
        data: np.array; all flux data
        labels: np.array; all data labels
        data_type: str; used to name the output files.
            Ex. data_type = 'Validation'
            Default = 'Train'
        
    Writes Out:
        The output .tfrecords file contains both flux data and labels.
        Output file is placed according to out_filename.
    '''
    
    ## Open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)  
    
    print('\nCommencing DataRecord creation..')
    
    for i in range(len(data)):
        ## Print how many TCEs are saved every 10 TCEs
        if (i % 10 == 0):
            print('{} Data: {}/{}'.format(data_type, i, len(data)))
            sys.stdout.flush()
        
        ## Print for last TCE
        elif (i == len(data)-1):
            print('{} Data: {}/{}'.format(data_type, i+1, len(data)))
            sys.stdout.flush()
        
        ## Load the TCE
        tce = data[i]
        
        ## Load the label
        label = labels[i]
        
        ## Quick check for empty TCEs
        if tce is None:
            continue

        ## Create a feature
        feature = {
            'flux_data': _bytes_feature(tce.tostring()),
            'label': _int64_feature(label.astype(int))
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()
    
    
def split_data(data, labels, train_size=0.8, val_size=0.1):
    '''
    The split_data function will split the data into three parts.
    These are training, validation, and testing, where each is
        also split by flux data and labels.
    
    Args: 
        data: np.array; all flux data
        labels: np.array; all labels
        train_size: Percentage of data to be split into training.
            Default = 0.8
        val_size: Percentage of data to be split into validation.
            Default = 0.1
        
    Returns:
        train_data: np.array; training flux data
        train_labels: np.array; training labels
        val_data: np.array; validation flux data
        val_labels: np.array; validation labels
        test_data: np.array; testing flux data
        test_labels: np.array; testing labels
    '''
    
    ## Split into training data
    train_data = data[0:int(train_size*len(data))]
    train_labels = labels[0:int(train_size*len(labels))]
    
    ## Ssplit into validation data
    val_data = data[int(train_size*len(data)):int((train_size+val_size)*len(data))]
    val_labels = labels[int(train_size*len(labels)):int((train_size+val_size)*len(labels))]
    
    ## Split into test data
    test_data = data[int((train_size+val_size)*len(data)):]
    test_labels = labels[int((train_size+val_size)*len(labels)):]
    
    return train_data,train_labels,val_data,val_labels,test_data,test_labels
    
    
def shuffle_data(data, labels, kepid_labels):
    '''
    The shuffle_data function will shuffle the order of the TCEs.
    
    Args: 
        data: np.array; hold all flux data
        labels: np.array; hold all labels
        kepid_labels: np.array; hold all kepids and labels
        
    Returns:
        data: shuffled np.array; hold all flux data
        labels: shuffled np.array; hold all labels
        kepid_labels: shuffled np.array; hold all 
            kepids and labels
    '''
    
    ## Zips up the data, shuffles, and unzips
    temp_list = list(zip(data, labels, kepid_labels))
    shuffle(temp_list)
    data, labels, kepid_labels = zip(*temp_list)
    
    ## Converts back into an np.array
    data = np.asarray(data)
    labels = np.asarray(labels)
    kepid_labels = np.asarray(kepid_labels)
    
    return data, labels, kepid_labels
    
    
################
## Main method for tfrecords creation
####
    
    
def main_tfrecords_creation(data, labels, kepid_labels, tf_dir='tfrecords'):
    '''
    The main_tfrecords_creation function will prepare the flux data and labels for the ML model.
    It does so by creating tfrecods files, which are optomized for tensorflow.
    
    Args: 
        data: np.array; all flux data
        labels: np.array; all labels
        kepid_labels: np.array; all kepids and labels
        tf_dir: str; directory to store the generated tfrecords
        
    Writes Out:
        The output .tfrecords files contains both flux data and labels.
        Output files are placed according to data type (eg. training, validation, testing).
        Output files are genereated using the create_data_record function.
        Output files are placed in tf_dir.
        
    '''
    
    ## Check if tfrecords directory already exist. If not, create it
    if os.path.isdir(tf_dir) == False:
        print('Directory {} does not exist.\nCreating tfrecords/'.format(tf_dir))
        os.mkdir(tf_dir)
    elif os.path.isdir(tf_dir) == True:
        print('Directory {} already exist.'.format(tf_dir))
        tf_dir = input('Enter new directory name: ')
        os.mkdir(tf_dir)
        
    ## Start counting towards time of completion
    start = time.time()
    
    ## Shuffling data
    print('\nShuffling data..')
    data, labels, kepid_labels = shuffle_data(data, labels, kepid_labels)
    
    ## Spliting data
    print('Splitting data..')
    train_data,train_labels,val_data,val_labels,test_data,test_labels = split_data(data, labels)
        
    ## Creating the train, validation, and test tfrecords
    create_data_record(tf_dir+'/'+'train.tfrecords', train_data, train_labels, 'Training')
    create_data_record(tf_dir+'/'+'val.tfrecords', val_data, val_labels, 'Validation')
    create_data_record(tf_dir+'/'+'test.tfrecords', test_data, test_labels, 'Testing')
    
    ## Display time of completion
    end = time.time()
    print('\nCompleted TFRecords creation in ' + str(round(end-start, 4)) + ' seconds')
    
    
