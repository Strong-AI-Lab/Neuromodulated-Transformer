import unittest

import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from models.attention_masks import *


class TestExternalMemory(unittest.TestCase):
    '''
    Class: TestExternalMemory
    Description: Class that performs unit tests on each class innvolved in the ExternalMemory computation.
    '''
    def test_CreatePaddingMask(self):
        x = tf.constant([[24, 33, 8, 0, 1], [16, 34, 8, 0, 0], [0, 0, 0, 9, 24]])

        y = create_padding_mask(x)

        self.assertEqual([y.shape[0], y.shape[1], y.shape[2], y.shape[3]],
                         [3, 1, 1, 5])

    def test_CreateLookAheadMask(self):
        seq_len = 8

        y = create_look_ahead_mask(seq_len)

        self.assertEqual([y.shape[0], y.shape[1]],
                         [seq_len, seq_len])

    def test_CreateMasks(self):

        batch, inp_seq_len, tar_seq_len = 4, 10, 12
        x = tf.ones((batch, inp_seq_len))
        y = tf.ones((batch, tar_seq_len))
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, y)

        self.assertEqual([enc_padding_mask.shape[0], enc_padding_mask.shape[1],
                          enc_padding_mask.shape[2], enc_padding_mask.shape[3]],
                         [batch, 1, 1, inp_seq_len])

        self.assertEqual([combined_mask.shape[0], combined_mask.shape[1],
                          combined_mask.shape[2], combined_mask.shape[3]],
                         [batch, 1, tar_seq_len, tar_seq_len])

        self.assertEqual([dec_padding_mask.shape[0], dec_padding_mask.shape[1],
                          dec_padding_mask.shape[2], dec_padding_mask.shape[3]],
                         [batch, 1, 1, inp_seq_len])



if __name__ == "__main__":
    unittest.main()