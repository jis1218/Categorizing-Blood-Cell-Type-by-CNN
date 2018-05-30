# coding: utf-8
'''
Created on 2018. 5. 29.

@author: Insup Jung
'''

import os
import tensorflow as tf
from numpy.core.tests.test_mem_overlap import xrange

def list_tfrecord_file(file_list):
    tfrecord_list = []
    for i in range(len(file_list)):
        current_file_abs_path = os.path.abspath(file_list[i])
        if current_file_abs_path.endswith(".tfrecord"):
            tfrecord_list.append(current_file_abs_path)
            print("Found %s successfully" % file_list[i])
        else :
            pass
        
    return tfrecord_list

def tfrecord_auto_traversal():
    current_folder_filename_list = os.listdir("./images/TRAIN/EOSINOPHIL") #Return a list containing the names of the files in the directory.
    if current_folder_filename_list != None :
        print("%s files were found under current folder. " %len(current_folder_filename_list))
        print("Please be noted that only files end with '*.tfrecord' will be load!")
        tfrecord_list = list_tfrecord_file(current_folder_filename_list)
        if len(tfrecord_list) !=0:
            for list_index in xrange(len(tfrecord_list)):
                #print(tfrecord_list[list_index])
                a = 1
        else:
            print("Cannot find any tfrecord files, please check the path.")
    
    return tfrecord_list

if __name__ == "__main__":
    tfrecord_list = tfrecord_auto_traversal()
        
    
    