import tensorflow as tf
import math


class CosineDecayLW(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr=0.0001, lower_bound_lr=0.00001, upper_bound_lr=0.01,
                 warmup_steps=2000, decay_steps=1000000, alpha=0.0):
        super(CosineDecayLW, self).__init__()

        assert start_lr > lower_bound_lr and start_lr < upper_bound_lr

        self.start_lr = start_lr
        self.lower_bound_lr = lower_bound_lr # not actually needed here.
        self.upper_bound_lr = upper_bound_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.alpha = alpha

        #warmup_lr_schedule = tf.linspace(self.start_lr, self.upper_bound_lr, self.warmup_steps)

        #iters = tf.range(decay_steps, dtype=tf.float32)
        #cosine_lr_schedule = tf.convert_to_tensor([self.decayed_learning_rate(i) for i in iters])
        #self.lr_schedule = tf.concat([warmup_lr_schedule, cosine_lr_schedule], axis=0)

    def __call__(self, step=2000):

        diff_warmup = abs(self.start_lr - self.upper_bound_lr) / self.warmup_steps

        initial_learning_rate = tf.convert_to_tensor(
            self.upper_bound_lr, name="initial_learning_rate")
        dtype = initial_learning_rate.dtype
        decay_steps = tf.cast(self.decay_steps, dtype)

        global_step_recomp = tf.maximum(tf.cast(step-self.warmup_steps, dtype), 1)
        global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
        completed_fraction = global_step_recomp / decay_steps
        cosine_decayed = 0.5 * (1.0 + tf.cos(
            tf.constant(math.pi, dtype=dtype) * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha
        #decayed_lr = tf.multiply(initial_learning_rate, decayed)

        return tf.cond(step < self.warmup_steps, lambda: tf.add(self.start_lr,diff_warmup*step),
                       lambda: tf.maximum(tf.multiply(initial_learning_rate, decayed), self.lower_bound_lr))
        #return self.lr_schedule[tf.cast(step, dtype=tf.dtypes.float64)[0]]
        #return self.lr_schedule

    # taken from https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
    def decayed_learning_rate(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.upper_bound_lr * decayed

if __name__ == "__main__":
    decay_schedule = CosineDecayLW(start_lr=0.0001, lower_bound_lr=0.000001, upper_bound_lr=0.01,
                                   warmup_steps=2000, decay_steps=3000*20)
    print([decay_schedule(i).numpy() for i in range(2500)])