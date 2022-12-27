from factory.Service import Service
from visualizator import dist_plot, countplot


def medical_insurance_prediction():
    factory_service = Service()
    service = factory_service.get_service()
    print(service)

    service.preprocess()

    service.build()
    service.test_accuracy_score()

    data = [31, 1, 25.74, 0, 1, 0]
    service.predict(data)

    df = service.get_dataset()
    # title = 'BMI Distribution'
    # dist_plot(df['bmi'], title)

    # countplot(df)

    # params = dict(x='children', data=df)
    # countplot(df, params)

    # params = dict(x='smoker', data=df)
    # countplot(df, params)

    title = 'Charges Distribution'
    dist_plot(df['charges'], title)

medical_insurance_prediction()






