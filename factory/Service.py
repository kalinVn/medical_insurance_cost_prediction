import config
from service.MedicalInsurancePrediction import MedicalInsurancePrediction
from service.MedicalInsurancePrediction import MedicalInsurancePrediction


class Service:

    def __init__(self):
        self.service_type = config.SERVICE_TYPE

    def get_service(self):
        if self.service_type == "ML":
            return MedicalInsurancePrediction()
        else:
            return MedicalInsurancePrediction()
