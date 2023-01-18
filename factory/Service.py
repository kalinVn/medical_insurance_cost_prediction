import config
from service.MedicalInsurancePrediction import MedicalInsurancePrediction
from service.MedicalInsurancePredictionKeras import MedicalInsurancePredictionKeras


class Service:

    def __init__(self):
        self.engine = config.ENGINE

    def get_service(self):

        if self.engine == "Sklearn":
            return MedicalInsurancePrediction()
        elif self.engine == "Keras":
            return MedicalInsurancePredictionKeras()
        else:
            return MedicalInsurancePrediction()
