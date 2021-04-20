import json

import numpy as np

from .basic import BasicMessage


class KerasMessage(BasicMessage):
    # _encoding = 'utf-8'

    def __init__(self, data=None):
        super().__init__(data)

    def to_json(self):
        orig_state = None
        for k, v in self.data.items():
            if k == self.STATE:
                orig_state = self.data[k]

                list_obj = list()

                for nd_arr in v:
                    list_obj.append(nd_arr.tolist())
                self.data[k] = list_obj

        json_data = json.dumps(self.data)

        # restore state in original format
        if orig_state:
            self.data[self.STATE] = orig_state

        return json_data

    @classmethod
    def to_object(cls, byte_data):
        if not byte_data:
            return None

        json_data = str(byte_data, cls._encoding)

        try:
            data = json.loads(json_data)
        except json.decoder.JSONDecodeError:
            return None

        if cls.STATE in data:
            list_obj = list()
            for arr in data[cls.STATE]:
                list_obj.append(np.asarray(arr))
            data[cls.STATE] = list_obj

        return cls(data)


if __name__ == "__main__":
    import tensorflow as tf
    from tensorflow import keras

    print(tf.version.VERSION)

    model = tf.keras.models.Sequential(
        [
            keras.layers.Dense(512, activation='relu', input_shape=(784, )),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10)
        ]
    )
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )

    print(f'initial: {model.get_weights()}')
    keras_data = KerasMessage()
    keras_data.set_state(model.get_weights())
    byte_data = keras_data.to_bytes()

    keras_data2 = keras_data.to_object(byte_data)

    print(f'keras_data = {keras_data}, keras_data2 = {keras_data2}')

    model2 = tf.keras.models.clone_model(model)
    model2.set_weights(keras_data2.get_state())

    print(f'after: {model2.get_weights()}')
