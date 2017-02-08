"""Reads directory of directories of images and converts images to vector representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import load_and_align_data
from rankings import Rankings
import argparse
import facenet
import json
import os
import sys
import tensorflow as tf
import time


def main(args):
    
    # TODO: Move constants to startup script
    ranker = Rankings(5, "../data/celeb_vectors.pk")

    with tf.Graph().as_default():
        with tf.Session() as sess:
          
            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            print('Watching: %s' % args.watch_dir)

            while True:
                # TODO: Only accept VALID (fully downloaded) image files
                image_filenames = os.listdir(args.watch_dir)
                if not image_filenames:
                    time.sleep(0.01)
                else:
                    # TODO: Implement max_batch_size
                    image_filepaths = [os.path.join(args.watch_dir, image_filename) for image_filename in image_filenames]
                    images = load_and_align_data(image_filepaths, args.image_size, args.margin, args.gpu_memory_fraction)

                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

                    # Run forward pass to calculate embeddings
                    feed_dict = { images_placeholder: images }
                    emb = sess.run(embeddings, feed_dict=feed_dict)

                    for filename, vector in zip(image_filenames, emb.tolist()):
                        image_filepath = os.path.join(args.watch_dir, filename)
                        output_filepath = os.path.join(args.output_dir, os.path.splitext(filename)[0] + ".csv")

                        rank = ranker.calculate_ranking(vector)
                        rank_names = [c.name for c in rank]
                        with open(output_filepath, 'w') as fp:
                            json.dump(rank_names, fp)

                        os.remove(image_filepath)
                
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model_dir', type=str, 
        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('watch_dir', type=str, help='Directory of images to process')
    parser.add_argument('output_dir', type=str, help='Output directory for vectors')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
