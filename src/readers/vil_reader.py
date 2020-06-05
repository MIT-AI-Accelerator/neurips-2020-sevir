import tensorflow as tf

class VILReader(tf.data.Dataset):
    def _generator(num_samples):
        # open file or assume that it's open already

        # read data
        for idx in num_samples:
            yield(idx)

    def __new__(cls, num_samples=4, output_shapes=(1,), dtype=tf.dtypes.int64):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=dtype,
            output_shapes=output_shapes,
            args=(num_samples,)
        )
