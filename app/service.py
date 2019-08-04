from app.core import prediction


def recognizer_service(input_data, model):
    try:
        data = input_data.get('pixels_metrix', [])
        response = prediction(data, model)
        return response
    except Exception as es:
        return {'result': [f'{es}'], 'message': 'some problem occured or input data is not as expected'}
