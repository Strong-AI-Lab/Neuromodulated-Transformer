'''
File name: ExternalMemory.py
Author: Kobe Knowles
Date created: 5/07/21
Data last modified: 5/07/21
Python Version: 3.6
Tensorflow version: 2
'''

import tensorflow as tf
import numpy as np

class ExternalMemory(tf.keras.layers.Layer):
    '''
    Class: ExternalMemory \n
    Description: Implementation of an External Memory component to be used in a transformer. \n
    Attributes:
        memory: (e.g. PrimitiveMemory)A supported memory class. \n
        memory_size: (int) The size of memory (i.e. its capacity). \n
        seq_len: (int) The sequence length of the tensors in memory (i.e. axis 1) \n
        d_model: (int) The dimension of the the transformer model, and the associated weights here that process it. \n
        recurrent_dim: (int) The dimension of the recurrent layers. \n
        rnn_forw: (tf.keras.layers.GRU|LSTM... etc.) The recurrent neural to be utilized by sentence attention with the input passed in the correct order.\n
        rnn_back: (tf.keras.layers.GRU|LSTM... etc.) The recurrent neural to be utilized by sentence attention with the input passed in reverse order.\n
        strategy: (str) The strategy to use when the class is called to extract information from the memory. \n
        wx: tf.keras.layers.Dense(d_model) Dense layer for word level attention on the input tensor. \n
        wc: tf.keras.layers.Dense(seq_len) Dense layer for word level attention on the memory. \n
        u: tf.keras.layers.Dense(recurrent_dim, activation='tanh') Dense layer to be used for `sentence' level attention. \n
        us: tf.keras.layers.Dense(1) Dense layer to be used for `sentence' level attention. \n
    '''

    def __init__(self, memory_type="primitive", memory_size=2, strategy="row_col_attn",
                 d_model=512, seq_len=512, recurrent_dim=512, recurrent_type="GRU", recurrent_dropout_rate=0.1):
        '''
        Function: __init__ \n
        Description: Initializer function for the ExternalMemory class. \n
        Input:
            memory_type: (str) Specifies the class to use as a memory. \n
            memory_size: (int) The size of the memory. \n
            strategy: (str) The strategy to use when the class is called to extract information from the memory. \n
            d_model: (int) The dimension of the the transformer model, and the associated weights here that process it. \n
            seq_len: (int) The sequence length of the tensors in memory (i.e. axis 1). \n
            recurrent_dim: (int) The dimension of the recurrent layers. \n
            recurrent_type: (str) The type of recurrent layers to use.
        '''
        super(ExternalMemory, self).__init__()

        if memory_type == "primitive":
            self.memory = PrimitiveMemory(memory_size) # todo add input... if needed later.
        #elif ...
        else:
            raise Exception(f"Invalid memory type: {memory_type}!")

        self.memory_size = memory_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.recurrent_dim = recurrent_dim

        self.rnn_forw = None
        self.rnn_back = None

        self.strategy = strategy
        if self.strategy == "row_col_attn":
            # Layers needed for word level attention.
            self.wx = tf.keras.layers.Dense(self.d_model)
            self.wc = tf.keras.layers.Dense(1)

            self.u = tf.keras.layers.Dense(self.recurrent_dim, activation='tanh')
            self.us = tf.keras.layers.Dense(1)
            if recurrent_type == "GRU":
                self.rnn_forw = tf.keras.layers.GRU(self.recurrent_dim, return_sequences=True,
                                                    go_backwards=False, dropout=recurrent_dropout_rate, return_state=True)
                self.rnn_back = tf.keras.layers.GRU(self.recurrent_dim, return_sequences=True, go_backwards=True,
                                                    dropout=recurrent_dropout_rate, return_state=True)
            elif recurrent_type == "LSTM":
                self.rnn_forw = tf.keras.layers.GRU(self.recurrent_dim, return_sequences=True, go_backwards=False,
                                                    dropout=recurrent_dropout_rate, return_state=True)
                self.rnn_back = tf.keras.layers.GRU(self.recurrent_dim, return_sequences=True, go_backwards=True,
                                                    dropout=recurrent_dropout_rate, return_state=True)
            else:
                raise Exception(f"Invalid recurrent type: {recurrent_type}!")

    def call(self, x):
        '''
        Function: call \n
        Description: Overrides parents class's call function (i.e. extract a vector from the external memory). \n
        Input:
            x: (tf.Tensor; (batch_size, seq_len, d_model)] Input tensor to attend to from memory in a context dependant manner. \n
        Return:
            (tf.Tensor; (batch_size, seq_len, d_model)])
        '''

        if self.memory.memory_capacity == 0: return tf.zeros((x.shape[0], x.shape[1], x.shape[2]))

        if self.strategy == "row_col_attn":
            return self._row_col_attn(x) # (batch_size, seq_len, d_model)
        else: raise Exception(f"Invalid stategy for ExternalMemory: {self.strategy}")


    def add_to_memory(self, x):
        '''
        Function: add_to_memory \n
        Description: Adds input tensor x to memory.
        Input:
            x: (tf.Tensor; [batch_size, seq_len, d_model]) Tensor to add to memory.
        Return:
            None
        '''
        self.memory.add(x)

    def _row_col_attn(self, x, training=True):
        '''
        Function: _row_col_attn \n
        Description: Implementation of hierarchical attention which first attends over the words in a memory slot,
            then over each memory slot to produce one final vector. \n
        Input:
            x: (tf.Tensor; (batch_size, seq_len, d_model)] Input tensor to attend to from memory in a context dependant manner.
        Return:
            (tf.Tensor; (batch_size, seq_len, d_model)])
        '''
        # TODO make sure gradient is turned off on the tensors in memory i.e. always applied to tf.stop_gradient.
        word_attn_memory = None # Will get to the following shape: (batch_size, memory_size, seq_len, d_model)
        # attend over each word
        for i in range(self.memory.memory_capacity): # note that order doesn't really matter because bidirectional GRU/LSTM/RNN used mostly.
            #print(f"i: {i}")
            #index = tf.convert_to_tensor([i])
            mem = self.memory.return_memory()
            z = self._word_attn(x, tf.stop_gradient(mem[i])) #self.memory(i) returns the tensor at index/slot i.
            #print("REACHHH")
            #z.shape = (batch_size, seq_len)
            z = tf.repeat(tf.expand_dims(z, -1), mem[i].shape[-1], axis=-1) # (batch_size, seq_len, d_model)
            z = z * tf.stop_gradient(self.memory(i)) # appy the attention weights to the memory.

            if word_attn_memory is None:
                word_attn_memory = tf.expand_dims(z, axis=1)
            else:
                word_attn_memory = tf.concat([word_attn_memory, tf.expand_dims(z, axis=1)], axis=1)
        segment_weights = self._sent_attn(word_attn_memory, training=training) # (batch_size, memory_size)

        if len(segment_weights.shape) == 1: # handles a memory size of 1
            segment_weights = tf.expand_dims(segment_weights, axis=1)
        # each memory slot is weighted
        out_tensor = None
        #print(f"word_attn_memory.shape: {word_attn_memory.shape}")
        #print(f"segment_weights.shape: {segment_weights.shape}")
        for i in range(word_attn_memory.shape[1]):
            mem_slot = tf.squeeze(word_attn_memory[:,i,:,:]) # (batch_size, seq_len, d_model)
            a = tf.repeat(tf.expand_dims(segment_weights[:, 0], -1), mem_slot.shape[1], axis=1)
            b = tf.repeat(tf.expand_dims(a, -1), mem_slot.shape[2], axis=2)  # (batch_size, seq_len, d_model)
            if out_tensor is None:
                out_tensor = mem_slot * b # (batch_size, seq_len, d_model)
            else:
                out_tensor = out_tensor + (mem_slot * b) # (batch_size, seq_len, d_model)

        return out_tensor # # (batch_size, seq_len, d_model)


    def _word_attn(self, x, c):
        '''
        Function: _word_attn \n
        Description: Implements word level attention on the memory cell/item slot. \n
        Input:
            x: (tf.Tensor; (batch_size, seq_len, d_model)] Input tensor to attend to from memory in a context dependant manner. \n
            c: (tf.Tensor; (batch_size, seq_len, d_model)] The memory to attend over (word level).
        Return:
            (tf.Tensor; (batch_size, seq_len)])
        '''

        x = self.wx(x) # (batch_size, seq_len, d_model)
        c = self.wc(tf.transpose(c, perm=[0,2,1])) # (batch_size, d_model, 1)
        attn = tf.matmul(x, c) # (batch, seq_len, 1)
        return tf.nn.softmax(tf.squeeze(attn)) # (batch, seq_len)

    def _sent_attn(self, c, training):
        '''
        Function: _sent_attn \n
        Description: Implements segment level attention over each slot in memory. \n
        Input:
            c: (tf.Tensor; (batch_size, memory_size, seq_len, d_model)] The memory to attend over (word level). \n
        Return:
            (tf.Tensor; (batch_size, memory_size)])
        '''
        # c.shape[1] is the memory_size or alternatively the number of columns.

        seq_output = None
        for b in range(c.shape[0]): # bidirectional class can only take 3 dimensions.
            #print(f"c.shape: {c.shape} \n {c[b,:,:,:].shape}")
            c_ = c[b,:,:,:]
            if len(c_.shape) == 2:
                c_ = tf.expand_dims(c_, axis=0) #  this only occurs when there is only one item in memory, so add this 3rd dimension back as it is removed because it is 1.

            seq_output_f, _ = self.rnn_forw(c_, training=training)  # (batch_size, memory_size, seq_len, rec_dim)
            seq_output_b, _ = self.rnn_back(c_, training=training) # (batch_size, memory_size, seq_len, rec_dim)
            seq_output_b  = tf.reverse(seq_output_b, [-2]) # reverse along the seq_len dimension.
            o = tf.concat([seq_output_f, seq_output_b], -1)  # (batch_size, memory_size, seq_len, rec_dim*2)
            if seq_output is None:
                seq_output = tf.expand_dims(o, axis=0)
            else:
                seq_output = tf.concat([seq_output, tf.expand_dims(o, axis=0)], axis=0)
        # seq_output.shape = (batch_size, memory_size, seq_len, rec_dim*2)


        seq_output = tf.reshape(seq_output, shape=(c.shape[0], c.shape[1], -1)) # (batch, memory_size, seq_len * (rec_dim*2))

        output = self.u(seq_output) # (batch_size, memory_size, rec_dim)
        output = self.us(output) # (batch_size, memory_size, 1)
        return tf.nn.softmax(tf.squeeze(output)) # (batch_size, memory_size)

class PrimitiveMemory(tf.keras.layers.Layer):
    '''
    Class: ExternalMemory \n
    Description: Implementation of an External Memory component to be used in a transformer. \n
    Attributes:
        memory_size: (int) The size of the memory. \n
        memory_capacity: (int) The current capacity of the memory. \n
        memory: (list) The current memory to store elements.
    '''
    def __init__(self, memory_size):
        '''
        Function: __init__ \n
        Description: Implementation of a primitive memory that operates the memory as a FIFO array. \n
        Input:
            memory_size: (int) The number of tensors (or elements) that can be stored in memory at once.
        '''
        super(PrimitiveMemory, self).__init__()
        # just have a list of vectors (tensors) as the memory.
        self.memory_size = memory_size
        self.memory_capacity = 0
        self.memory = list() # list instead of a set because order matters and is faster to iterate over the list.

    def call(self, index):
        '''
        Function: call \n
        Decstiption: Returns the memory at the given index. \n
        Input:
            index: (int) The memory index to retrieve. --- change to tensor...
        Return:
            (tf.Tensor; [batch_size, seq_len, d_model]) most commonly.
        '''
        print(f"\n\nindex: {tf.reshape(index, shape=(1))}\n\n")
        return self.memory[index[0]]

    def return_memory(self):
        return self.memory

    def add(self, x):
        '''
        Function: add \n
        Description: Adds a tensor to the start of memory. \n
        Input:
            x: (tf.Tensor; [batch_size, seq_len, d_model]) most commonly.
        '''
        if self.memory_capacity < self.memory_size:
            self.memory.insert(0, x)
            self.memory_capacity += 1
        else:
            self.memory.insert(0, x) # insert into the first element.
            self.memory.pop() # removes the last element.
            # memory_capacity stays the same here.

    def remove(self, index=-1):
        '''
        Function: remove \n
        Description: Remove an item from memory at a certain index. \n
        Input:
            index: (int) Position to remove item from memory.
        '''
        self.memory.pop(index)
        self.memory_capacity -= 1

    def reset(self):
        self.memory = list() # i.e. reset the memory.
        self.memory_capacity = 0

if __name__ == "__main__":
    batch_size = 2
    seq_len = 32
    d_model = 12
    em = ExternalMemory(memory_type="primitive", memory_size=3, strategy="row_col_attn",
                 d_model=d_model, seq_len=seq_len, recurrent_dim=512, recurrent_type="GRU")


    for i in range(4):
        x = tf.ones((batch_size, seq_len, d_model)) * (i+1)
        em.add_to_memory(x) # to 4 to test if the pop works as intended.
    #print(f"Currently in memory: {em.memory.memory}")
    inp_ = tf.random.uniform((batch_size, seq_len, d_model))

    tens = em(inp_)
    print(f"The shape of the output tensor: {tens.shape}")
