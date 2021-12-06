import tensorflow as tf
import numpy as np

def init_vanilla_ffn(d_model: int, dff: int):
    '''
    Description: Initializes the feed-forward network for a transformer layer. \n
    :param d_model: (int) An integer specifying the dimension of the transformer model.
    :param dff: (int) An integer specifying the dimension of the feed-forward layer.
    :return: (tf.keras.Sequential)
    '''
    return tf.keras.Sequential([
        #tf.keras.layers.Dense(dff, activation='relu', input_shape=(d_model,)),
        tf.keras.layers.Dense(dff, activation='gelu', input_shape=(d_model,)),
        tf.keras.layers.Dense(d_model, activation=None, input_shape=(dff,))
    ])

class FeedForwardNetwork(tf.keras.layers.Layer):
    '''
    Description: Implementation of the feed-forward layer of a transformer.
    '''

    def __init__(self, d_model: int, dff: int, type: str):
        '''
        :param d_model: (int) An integer specifying the dimension of the transformer model.
        :param dff: (int) An integer specifying the dimension of the feed-forward layer.
        :param type: (str) A string specifying the type of feed-forward layer.
        '''
        super(FeedForwardNetwork, self).__init__()

        self.d_model = d_model # dimension of the transformer.
        self.dff = dff # dimension of the feed forward layer.

        self.ffn = None
        if type == "vanilla" or type == "default":
            self.ffn = init_vanilla_ffn(d_model, dff)
        else: raise Exception(f"Invalid value for type ({type})!")

    def call(self, x):
        '''
        :param x: (tf.Tensor; [batch_size, seq_len, d_model]) A tensor specifying the input to the model.
        :return: (tf.Tensor; [batch_size, seq_len, d_model]
        '''
        assert len(x.shape) == 3, f"Invalid input shape dimension. There should only be 3, got {len(x.shape)}!"

        return self.ffn(x)

if __name__ == "__main__":
    d_model, max_seq_len, dff, batch_size = 10, 8, 20, 4
    ffn = FeedForwardNetwork(d_model, dff, type="vanilla")

    inp =  tf.random.uniform((batch_size, max_seq_len, d_model))
    tens = ffn(inp)

    print(f"ffn output tensor: {tens} \n"
          f"shape: {tens.shape}")
