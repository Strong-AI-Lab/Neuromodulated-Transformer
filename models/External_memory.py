import tensorflow as tf
import numpy as np

class ExternalMemory(tf.keras.layers.Layer):
    '''
    Class: ExternalMemory
    Description: Parent memory class that implements external memory depending on the passed parameters.
    Input:
        nrow: (int) The number of rows in the memory matrix (how many memories to store)
        ncol: (int) The numnber of columns in the embedding matrix.
        col_dim: (int) The dimension of the word embeddings which are to be stored one per column and row.
        mem_type: (string) Represents the type of memory to implement. Each string references a
            specific class to call.
        strategy: (string)  Represents the type of strategy to implement when calling the memory module.
    '''
    def __init__(self, nrow, ncol, col_dim, rec_dim, mem_type="Default",
                 strategy="Hierarchical Attention", recurrent_model="GRU"):
        super(ExternalMemory, self).__init__()

        self.nrow = nrow
        self.ncol = ncol
        self.w_emb = col_dim
        self.mem_type = mem_type
        self.strat = strategy
        self.recurrent_model = recurrent_model
        self.rec_dim = rec_dim

        self.memory = None
        if self.mem_type == "Default":
            # This represents a simple matrix implementation of memory
            self.memory = MatrixMemory(self.nrow, self.ncol, self.w_emb)

        assert self.memory is not None

        self.strategy = None
        if self.strat == "Hierarchical Attention":
            # Class that performs hierarchical attention on columns (words) then rows (sentences)
            self.strategy = HierarchicalAttention(self.nrow, self.ncol, self.w_emb,
                                                  self.recurrent_model, self.rec_dim)

        assert self.strategy is not None


    def call(self, x):
        '''
        Function: call
        Description: Override original call function and return x with added information from memory via
            word and sentence attention.
        Inputs:
            x: (tf.Tensor; (batch, seq_len, d_model)) Output from a transformer layer.
        Return:
            output: (tf.Tensor ;(batch, seq_len|ncol, w_dim))
        '''
        #batch_attn_masks = []
        mask = tf.Variable(tf.zeros((x.shape[0], self.nrow, self.ncol))) # (batch, nrow, ncol)
        for row in range(self.nrow):
            ci_attn = self.strategy.calc_word_attn(x, tf.squeeze(self.memory([row, row+1], [0, self.ncol]))) # (batch, ncol|n)
            ci_attn = tf.expand_dims(ci_attn, axis=1) # (batch, 1, ncol)
            #mask[:, row, :] = ci_attn
            mask = mask[:, row:row+1, :].assign(ci_attn) # row:row+1 so the ncol|seq_len dimension is kept as 1.

        #output = tf.zeros(x.shape[0], self.ncol, self.w_emb) # (batch, seq_len, word_dim)
        output = None
        ememory = self.memory([0, self.nrow], [0, self.ncol]) # (nrow, ncol, w_dim)
        for b in range(x.shape[0]):
            curr_batch_mask = tf.expand_dims(mask[b, :, :], axis=2) # (nrow, ncol, 1)
            curr_batch_mask = tf.tile(curr_batch_mask, [1, 1, ememory.shape[2]]) # (nrow, ncol, w_dim)
            assert curr_batch_mask.shape == ememory.shape
            masked_memory = curr_batch_mask * ememory
            alpha = self.strategy.calc_sent_attn(masked_memory) # nrow
            alpha = tf.tile(tf.expand_dims(tf.expand_dims(alpha, axis=1), axis=2), [1, self.ncol, self.w_emb]) # (nrow, ncol, w_dim)
            v = alpha * masked_memory # (nrow, ncol, w_dim)
            v = tf.expand_dims(tf.reduce_sum(v, axis=[0]), axis=0) # (1, ncol|seq_len, w_dim)
            if output is None:
                output = v # (1, ncol, w_dim)
            else:
                output = tf.concat([output, v], 0) # (b+1, ncol, w_dim)

        return output # (batch, seq_len|ncol, w_dim)

    def add_to_memory(self, x, strategy="Default"):
        '''
        Function: add_to_memory
        Description: Adds the input x to memory.
        Inputs:
            x: (batch, seq_len, d_model) Tensor to add to memory.
            strategy: (string) To be used to select what strategy to use to add x to memory.
        Return:
            bool_: (boolean) Either True or False indicating if x has been added to memory successfully.
        '''
        bool_ = self.memory.add(x, strategy)
        return bool_

class MatrixMemory(tf.keras.layers.Layer):
    '''
    Class: MatrixMemory
    Description: Implementation of a matrix memory to be used with the ExternalMemory class.
    Inputs:
        nrow: (int) The number of rows in the memory matrix (how many memories to store)
        ncol: (int) The numnber of columns in the embedding matrix.
        word_dim: (int) The dimension of the word embeddings which are to be stored one per column and row.
    '''
    def __init__(self, nrow, ncol, word_dim):
        super(MatrixMemory, self).__init__()

        self.nrow = nrow
        self.ncol = ncol
        self.word_dim = word_dim

        self.weight_matrix = tf.Variable(tf.zeros((self.nrow, self.ncol, self.word_dim), dtype=tf.float32))
        self.max_counter = 0
        self.counter = 0

    def call(self, row, col):
        '''
        Function: Call
        Description: Overrides original
        Input:
            row: (list) list[0] is where to start row indexing; list[1] is where to end row indexing.
            col: (list) list[0] is where to start col indexing; list[1] is where to end col indexing.
        '''
        assert row[0] >= 0, "row[0] can't be negative."
        assert row[1] <= self.weight_matrix.shape[0], "row[1] can't be greater than the number of rows in memory."
        assert row[0] < row[1], "row[0] must be strictly less than row[1]."
        assert col[0] >= 0, "col[0] can't be negative."
        assert col[1] <= self.weight_matrix.shape[1], "col[1] can't be greater than the number of columns."
        assert col[0] < col[1], "col[0] must be strictly less than col[1]."

        return self.weight_matrix[row[0]:row[1], col[0]:col[1], :] # becasue :, :, : at each position, the shape stays 3D.

    def add(self, x, strategy="default"):
        '''
        Function: add
        Description: Add the input to memory according to the specified strategy.
        Input:
            x: (tf.Tensor; (batch, n_col|n, d_model)
            strategy: (string) Represents the strategy to use to update memories.
        Return:
            (boolean) Either True or False indicating if x has been added to memory successfully.
        '''

        try:
            assert x.shape[1] == self.ncol and x.shape[2] == self.word_dim, "Input dimensions don't match what is expected"

            batch = x.shape[0]


            if self.counter + batch <= self.nrow:
                start = self.counter
                end = self.counter + batch
                self.weight_matrix = self.weight_matrix[start:end, :, :].assign(x)
                self.counter = end
            else:
                # first nrow-1 - start are added here.
                start = self.counter # start:nrow
                '''
                end = c-1 + batch - (nrow-1)
                    = c + batch - nrow
                '''
                end =  (self.counter + batch) - (self.nrow) # 0:end
                # Split up batch into two (uneven) halves.
                # 0:nrow-start is the first half.
                self.weight_matrix = self.weight_matrix[start:, :, :].assign(x[:self.nrow-start, :, :])
                # nrow-start: is the second half.
                self.weight_matrix = self.weight_matrix[:end, :, :].assign(x[self.nrow-start:, :, :])
                self.counter = end

            # Keep track of counter to know how much of memory has been utilized.
            self.max_counter += batch
            self.max_counter = min(self.nrow, self.max_counter)

            return True

        except:
            return False




class HierarchicalAttention(tf.keras.layers.Layer):
    '''
    Class HierarchicalAttentionMemory
    Description: Implementation of the memory itself, with functions to perform
        attention at a word level, then at a sentence level (hierarchical attention).
    Input:
        nrow: (int) Number of rows in memory. The number of elements in memory.
        ncol: (int) Number of columns in memory. The length of memory.
        word_dim: (int) Dimension of the word embeddings.
        recurrent_model: (string) What recurrent architecture to use for sentence attention.
            "LSTM": Vanilla LSTM.
            "GRU":  Vanilla GRU.
        recurrent_dim: (int) Dimension for the chosen recurrent architecture.
    '''

    def __init__(self, nrow, ncol, word_dim, recurrent_model, recurrent_dim):
        super(HierarchicalAttention, self).__init__()

        self.nrow = nrow
        self.ncol = ncol
        self.word_dim = word_dim

        self.recurrent_model = recurrent_model
        self.recurrent_dim = recurrent_dim


        # No longer needed.
        #assert self.recurrent_dim == self.word_dim, "Word dimension needs to match the recurrent dimension."

        self.bdir_recurrent = None
        if self.recurrent_model == "GRU":
            # return_sequences: True --> Return the full sequence.
            # return_state: True --> Return last state in addition to the output.
            recurrent = tf.keras.layers.GRU(self.recurrent_dim,
                                            return_sequences=True)
                                                         #return_state=True) # Do I need the hidden states?

            self.bdir_recurrent = tf.keras.layers.Bidirectional(recurrent)
        elif self.recurrent_model == "GRU":
            recurrent = tf.keras.layers.LSTM(self.recurrent_dim,
                                            return_sequences=True)
                                                         #return_state=True) # Do I need the hidden states?

            self.bdir_recurrent = tf.keras.layers.Bidirectional(recurrent)

        assert self.bdir_recurrent is not None, "Error when initializing bidirectinal recurrent model. "


        self.wx = tf.keras.layers.Dense(self.word_dim)
        self.wc = tf.keras.layers.Dense(1) # input will be of dimension ncol.

        self.u = tf.keras.layers.Dense(self.recurrent_dim, activation="tanh") # input dim is (ncol*(word_dim*2))
        self.us = tf.keras.layers.Dense(1) # W = (recurrent_dim, 1)

    def calc_word_attn(self, x, c):
        '''
        Function: calc_word_attn
        Description: Calculates attention score for each word in a single memory row for all elements in a batch.
        Input:
            x: (tf.Tensor; (batch, ncol|n, word_dim))
            c: (tf.Variable(tf.Tensor); (ncol, word_dim))
        Return:
            attn: (tf.Tensor; batch, ncol|n) Returns attention scores for current sentence/segement.
        '''
        #TODO: Add dropout layers.
        x = self.wx(x) # (batch, ncol, word_dim)
        c_ = self.wc(tf.transpose(c, perm=[1,0])) # (word_dim, 1)

        attn = tf.matmul(x, c_) # (batch, ncol|n, 1)

        return tf.nn.softmax(tf.squeeze(attn)) # (batch, ncol|n)

    def calc_sent_attn(self, c):
        '''
        Function: calc_sent_attn
        Description: Calculates the attention score for each sentence and returns it.
            Notice that this is done one batch at a time in this implementation.
        Input:
            c: (tf.Tensor; (nrow, ncol, word_dim))
                Attention adjusted memory for a given batch (each batch will have a different score).
                Note: This is done one batch at a time.
        Return:
            alpha: (tf.Tensor; (nrow))
                The attention scores for a single batch are returned.
        '''
        #TODO: Add dropout layers.
        if self.recurrent_model == "GRU":
            seq_output = self.bdir_recurrent(c) # (nrow, ncol, rec_dim*2)

        seq_output = tf.reshape(seq_output, shape=(self.nrow, -1)) # Note: This will work as the input dimension will always be the same.
        # (nrow, ncol*rec_dim*2)

        assert seq_output.shape[1] == self.ncol * (self.recurrent_dim*2), "Invalid tensor dimensions from reshape"

        output = self.u(seq_output) # (nrow, rec_dim)
        output = self.us(output) # (nrow, 1)
        alpha = tf.nn.softmax(tf.squeeze(output)) # (nrow)

        return alpha # (nrow)


if __name__ == "__main__":
    pass



