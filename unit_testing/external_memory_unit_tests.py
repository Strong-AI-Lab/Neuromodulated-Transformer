import unittest

import sys
sys.path.append("..")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from models.External_memory import *


class TestExternalMemory(unittest.TestCase):
    '''
    Class: TestTransformerShapes
    Description: Class that performs unit tests on each Transformer layer. Each test involves
        testing if the shape of the output of each module matches what is expected.

    https://www.tensorflow.org/tutorials/text/transformer is where the test cases/code is taken from.
    '''
    def test_ExternalMemoryInitialization(self):
        ememory = ExternalMemory(nrow=10, ncol=20, col_dim=50, rec_dim=50, mem_type="Default",
                                 strategy="Hierarchical Attention", recurrent_model="GRU")
        #self.assertEqual(item1, item2)

    def test_ExternalMemoryCall(self):
        batch, seq_len, dim, nrow, ncol = 16, 8, 100, 10, 8

        ememory = ExternalMemory(nrow=nrow, ncol=ncol, col_dim=dim, rec_dim=dim, mem_type="Default",
                                 strategy="Hierarchical Attention", recurrent_model="GRU")

        x = tf.random.uniform([batch, seq_len, dim])

        output = ememory(x)
        self.assertEqual([output.shape[0], output.shape[1], output.shape[2]],
                         [batch, seq_len, dim])

    def test_ExternalMemoryAdd(self):
        batch, seq_len, dim, nrow, ncol = 16, 8, 100, 10, 8
        ememory = ExternalMemory(nrow=nrow, ncol=ncol, col_dim=dim, rec_dim=dim, mem_type="Default",
                                 strategy="Hierarchical Attention", recurrent_model="GRU")

        x = tf.random.uniform([batch, ncol, dim])

        bool_ = ememory.add_to_memory(x)

        self.assertEqual(True, bool_)

    def test_MatrixMemoryInitialization(self):
        batch, seq_len, dim, nrow, ncol = 16, 8, 100, 10, 8
        mmemory = MatrixMemory(nrow=nrow, ncol=ncol, word_dim=dim)

    def test_MatrixMemoryCall(self):
        batch, seq_len, dim, nrow, ncol = 16, 8, 100, 10, 8
        mmemory = MatrixMemory(nrow=nrow, ncol=ncol, word_dim=dim)

        # Test for the correct shapes.
        m = mmemory([0, nrow], [0, seq_len])
        self.assertEqual([m.shape[0], m.shape[1], m.shape[2]],
                         [nrow, seq_len, dim])

        m = mmemory([0, nrow-2], [0, seq_len-3])
        self.assertEqual([m.shape[0], m.shape[1], m.shape[2]],
                         [nrow-2, seq_len-3, dim])

        m = mmemory([2, 7], [3, seq_len-2])
        self.assertEqual([m.shape[0], m.shape[1], m.shape[2]],
                         [5, 3, dim])

        m = mmemory([0, 1], [0, seq_len])
        self.assertEqual([m.shape[0], m.shape[1], m.shape[2]],
                         [1, seq_len, dim])

        # Test row[0] < row[1]
        with self.assertRaises(AssertionError) as a:
            mmemory([0, 0], [0, 3])

        # Test row[0] < row[1]
        with self.assertRaises(AssertionError) as a:
            mmemory([2, 0], [0, 3])

        # Test col[0] < col[1]
        with self.assertRaises(AssertionError) as a:
            mmemory([0, 2], [0, 0])

        # Test col[0] < col[1]
        with self.assertRaises(AssertionError) as a:
            mmemory([0, 2], [2, 0])

        # test row[1] <= nrow
        with self.assertRaises(AssertionError) as a:
            mmemory([0, 240], [0, 3])

        # test col[1] <= ncol
        with self.assertRaises(AssertionError) as a:
            mmemory([0, 2], [0, 240])

        # test col[0] >= 0
        with self.assertRaises(AssertionError) as a:
            mmemory([0, 2], [-2, 3])

        # test row[0] >= 0
        with self.assertRaises(AssertionError) as a:
            mmemory([-2, 2], [0, 3])



    def test_MatrixMemoryAdd(self):
        batch, seq_len, ncol, dim, nrow = 16, 8, 8, 100, 20

        mmemory = MatrixMemory(nrow=nrow, ncol=seq_len, word_dim=dim)
        x = tf.ones([batch, seq_len, dim]) * 3

        mmemory.add(x=x, strategy='default')
        mat1 = tf.concat([tf.ones([batch, ncol, dim]) * 3, tf.zeros([nrow - batch, ncol, dim])], axis=0)
        # print("\nMat1", mat1.shape, "\n")
        # print("\nMemmatrix", mmemory.weight_matrix.shape, "\n")
        # print(tf.equal(mmemory.weight_matrix, mat1))
        self.assertTrue(tf.equal(mmemory.weight_matrix, mat1).numpy().all())
        self.assertTrue(mmemory.max_counter == batch)
        self.assertTrue(mmemory.counter == batch)

        y = tf.ones([batch, seq_len, dim]) * 2
        mmemory.add(x=y, strategy='default')
        mat2 = tf.concat([tf.ones([12, ncol, dim]) * 2, tf.ones([4, ncol, dim]) * 3,
                          tf.ones([4, ncol, dim]) * 2], axis=0)
        # print("\n", mmemory.weight_matrix[13,:,:], "\n")
        self.assertTrue(tf.equal(mmemory.weight_matrix, mat2).numpy().all())
        self.assertTrue(mmemory.max_counter == nrow)
        self.assertTrue(mmemory.counter == 12)

        batch = 2
        mmemory = MatrixMemory(nrow=nrow, ncol=seq_len, word_dim=dim)

        x = tf.ones([batch, seq_len, dim])
        mmemory.add(x, strategy='default')
        mmemory.add(x, strategy='default')
        mat3 = tf.concat([tf.ones([4, ncol, dim]), tf.zeros([16, ncol, dim])], axis=0)
        self.assertTrue(tf.equal(mmemory.weight_matrix, mat3).numpy().all())
        self.assertTrue(mmemory.max_counter == 4)
        self.assertTrue(mmemory.counter == 4)

    def test_HierarchicalAttentionInitialization(self):
        batch, seq_len, ncol, dim, nrow = 16, 8, 8, 100, 20

        #mmemory = MatrixMemory(nrow=nrow, ncol=seq_len, word_dim=dim)
        hattn = HierarchicalAttention(nrow=nrow, ncol=ncol, word_dim=dim, recurrent_model='GRU', recurrent_dim=dim)


    def test_HierarchicalAttentionWordAttn(self):
        batch, seq_len, ncol, dim, nrow = 16, 8, 8, 100, 20

        mmemory = MatrixMemory(nrow=nrow, ncol=seq_len, word_dim=dim)
        hattn = HierarchicalAttention(nrow=nrow, ncol=ncol, word_dim=dim, recurrent_model='GRU', recurrent_dim=dim)
        x = tf.random.uniform((batch, seq_len, dim), dtype=tf.float32)

        attn_weights = hattn.calc_word_attn(x, tf.squeeze(mmemory([2,3], [0,ncol])))

        # There should only be two dimensions.
        self.assertTrue(len(attn_weights.shape) == 2)
        # The two dimensions should match [batch, ncol]
        self.assertEqual([attn_weights.shape[0], attn_weights.shape[1]],
                         [batch, ncol])
        # each batch should equal 1.
        self.assertTrue(tf.equal(tf.reduce_sum(attn_weights, 1), tf.ones((batch))).numpy().all())
        #print(attn_weights)

    def test_HierarchicalAttentionSentAttn(self):
        batch, seq_len, ncol, dim, nrow, rec_dim = 16, 8, 8, 100, 20, 120

        mmemory = MatrixMemory(nrow=nrow, ncol=seq_len, word_dim=dim)
        hattn = HierarchicalAttention(nrow=nrow, ncol=ncol, word_dim=dim, recurrent_model='GRU', recurrent_dim=rec_dim) #recurrent_dim=112) error as reccurrent_dim has to equ
        x = tf.random.uniform((batch, seq_len, dim), dtype=tf.float32)

        attn_weights = hattn.calc_sent_attn(mmemory([0, nrow], [0, ncol]))

        # 1D object test.
        self.assertTrue(len(attn_weights.shape) == 1)
        # The dimension should be of size nrow
        self.assertEqual(attn_weights.shape[0], nrow)
        # All elements should sum to one.
        self.assertTrue(tf.equal(tf.reduce_sum(attn_weights), tf.ones((1))).numpy().all())

        hattn = HierarchicalAttention(nrow=nrow, ncol=ncol, word_dim=dim, recurrent_model='GRU',
                                      recurrent_dim=dim)  # recurrent_dim=112) error as reccurrent_dim has to equ
        x = tf.random.uniform((batch, seq_len, dim), dtype=tf.float32)

        attn_weights = hattn.calc_sent_attn(mmemory([0, nrow], [0, ncol]))

        # 1D object test.
        self.assertTrue(len(attn_weights.shape) == 1)
        # The dimension should be of size nrow
        self.assertEqual(attn_weights.shape[0], nrow)
        # All elements should sum to one.
        self.assertTrue(tf.equal(tf.reduce_sum(attn_weights), tf.ones((1))).numpy().all())

if __name__ == "__main__":
    unittest.main()