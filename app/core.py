import numpy as np


def prediction(input_sample, model):

    data = input_sample.reshape(1, 28, 28, 1)
    prob = model.predict(data)
    pred = np.argmax()
    result = f'LABEL PREDICTED WITH PROBABILITY {prob[pred]} IS {pred}'
    result_object = {'result': result}
    return result_object
