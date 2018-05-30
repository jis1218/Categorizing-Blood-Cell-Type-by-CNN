# coding: utf-8
'''
Created on 2018. 5. 30.

@author: Insup Jung
'''

from __future__ import absolute_import

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
from builtins import int
from numpy.core.tests.test_mem_overlap import xrange

tf.app.flags.DEFINE_string('labels_file', './label.txt', 'Labels file')

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):

    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_example(filename, image_buffer, label, text, height, width):
    
    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height':_int64_feature(height),
        'image/width' : _int64_feature(width),
        'image/colorspace' : _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels' : _int64_feature(channels),
        'image/class/label' : _int64_feature(label),
        'image/class/text' : _int64_feature(tf.compat.as_bytes(text)),
        'image/format' : _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename' : _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded' : _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example

class ImageCoder(object):
    
    def __init__(self):
        self._sess = tf.Session()
        
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        
    def decode_jpeg(self, image_data): #image_data를 넣어주면 jpeg를 decode 한다.
        image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data : image_data})
        assert len(image.shape) ==3
        assert image.shape[2] == 3
        return image

def _process_image(filename, coder):
    
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()
        
    image = coder.decode_jpeg(image_data)
    
    height = image.shape[0]
    width = image.shape[1]
    
    return image_data, height, width

def _process_image_files_batch(coder, thread_index, ranges, name, filenames, texts, labels, num_shards):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards/ num_threads)
    
    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    
    counter = 0
    
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.2d-of-%.2d.tfrecord' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
    
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            text = texts[i]
    
            image_buffer, height, width = _process_image(filename, coder)
    
            example = convert_to_example(filename, image_buffer, label,
                                        text, height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            print(counter)
    
            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                  (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()
    
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
            (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.
    
      Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
      """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)
    
    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])
    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()
    
    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    
    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()
    
    threads = []
    for thread_index in xrange(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)
    
    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
            (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the image data set resides in JPEG files located in
      the following directory structure.

        data_dir/dog/another-image.JPEG
        data_dir/dog/my-image.jpg

      where 'dog' is the label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
  """
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
      labels_file, 'r').readlines()]

    labels = []
    filenames = []
    texts = []

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of JPEG files and labels.
    for text in unique_labels:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

    if not label_index % 100:
        print('Finished finding files in %d of %d classes.' % (
          label_index, len(labels)))
    label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(unique_labels), data_dir))
    return filenames, texts, labels


def _process_dataset(name, directory, num_shards, labels_file):
    """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
    filenames, texts, labels = _find_image_files(directory, labels_file)
    _process_image_files(name, filenames, texts, labels, num_shards)


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
    assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
    print('Saving results to %s !' % FLAGS.output_directory)

    # Run it!
    _process_dataset('validation', FLAGS.validation_directory,
                   FLAGS.validation_shards, FLAGS.labels_file)
    _process_dataset('train', FLAGS.train_directory,
                   FLAGS.train_shards, FLAGS.labels_file)


if __name__ == '__main__':
    tf.app.run()
        
        