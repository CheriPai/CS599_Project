"""Reads directory of directories of images and converts images to vector representation."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
from utils import load_and_align_data
import argparse
import facenet
import json
import os
import sys
import tensorflow as tf


def main(args):

    output = {}

    with tf.Graph().as_default():
        with tf.Session() as sess:
          
            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)

            for i, celeb in enumerate(tqdm(os.listdir(args.image_dir))):
                celeb_path = os.path.join(args.image_dir, celeb)
                image_files = [os.path.join(celeb_path, f) for f in os.listdir(celeb_path)]
                images = load_and_align_data(image_files, args.image_size, args.margin, args.gpu_memory_fraction)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images }
                emb = sess.run(embeddings, feed_dict=feed_dict)

                output[celeb] = emb.tolist()

    with open('celeb_vectors.json', 'w') as fp:
        json.dump(output, fp)
                
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('image_dir', type=str, help='Directory of images to process')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
