import pandas as pd
import numpy as np
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')

with open('model.pkl', 'br') as file:
    model = pkl.load(file)
    
with open('encodings.pkl', 'br') as file:
    encodings = pkl.load(file)

COLUMNS = ['manufacturer', 'fuel', 'type', 'year', 'odometer', 'long', 'car_age']

BRANDS = {
    'gmc': 'Mainstream',
    'chevrolet': 'Mainstream',
    'toyota': 'Mainstream',
    'ford': 'Mainstream',
    'jeep': 'Mainstream',
    'nissan': 'Mainstream',
    'ram': 'Mainstream',
    'mazda': 'Mainstream',
    'cadillac': 'Premium',
    'honda': 'Mainstream',
    'dodge': 'Mainstream',
    'lexus': 'Premium',
    'jaguar': 'Premium',
    'buick': 'Mainstream',
    'chrysler': 'Mainstream',
    'volvo': 'Premium',
    'audi': 'Premium',
    'infiniti': 'Premium',
    'lincoln': 'Premium',
    'alfa-romeo': 'Premium',
    'subaru': 'Mainstream',
    'acura': 'Premium',
    'hyundai': 'Mainstream',
    'mercedes-benz': 'Premium',
    'bmw': 'Premium',
    'mitsubishi': 'Mainstream',
    'volkswagen': 'Mainstream',
    'porsche': 'Premium',
    'kia': 'Mainstream',
    'rover': 'Mainstream',
    'ferrari': 'Luxury',
    'mini': 'Mainstream',
    'pontiac': 'Mainstream',
    'fiat': 'Mainstream',
    'tesla': 'Premium',
    'saturn': 'Mainstream',
    'mercury': 'Mainstream',
    'harley-davidson': 'Mainstream',
    'aston-martin': 'Luxury',
    'land rover': 'Premium',
    'morgan': 'Luxury'
}

def getUserInput():
    print('Si no sabes el dato, no escribas nada')
    print('='*100)
    array = []
    questions = ['¿En qué región quieres vender o comprar tu coche?',
                 '¿Cuál es la marca de fabricante del coche?',
                 '¿Qué tipo de combustible tiene el coche? (gas, diesel, hybrid, electric, other)',
                 '¿Cuál es el estado del coche? (clean, rebuilt, salvage, missing, lien, parts only)',
                 '¿Qué tipo de transmisión tiene el coche? (automatic, manual, other)',
                 '¿Que tipo de vehículo es? (sedan, SUV, bus, truck, pickup, coupe, hatchback, wagon, van, convertible, mini-van, offroad, other)',
                 '¿En que estado de EEUU quieres vender o comprar tu coche?',
                 '¿De qué color es el coche?',
                 '¿En qué condición está el coche? (new, like new, excellent, good, fair, salvage)',
                 '¿Cuántos cilindros tiene el motor?',
                 '¿Qué transmisión tiene? (4wd, rwd, fwd)',
                 '¿Cuándo se fabricó?',
                 '¿Cuántos kilómetros tiene?',
                ]
    for idx, question in enumerate(questions):
        print('-'*100)
        data = input(question)
        if data == '':
            array.append(np.nan)
            continue
            
        if idx == 9:
            array.append(data + ' cylinders')
            continue
        
        if idx >= 11:
            array.append(float(data))
            continue
        
        array.append(data)
        
        
    array.append(39.1501)
    array.append(-88.4326)
    
    return array


def generateCols(array):
    new_array = array.copy()
    manufacturer = array[1]
    year = array[-4]
    new_array.insert(7, BRANDS.get(manufacturer))
    
    new_array.append(2021-year)
    
    return new_array



def applyEncodings(array, encodings):
    dimentions = ['region', 'manufacturer', 'fuel', 'title_status', 'transmission', 'type', 'state', 'brand_segmentation', 'paint_color', 'condition', 'cylinders', 'drive', 'year', 'odometer', 'lat', 'long', 'car_age']
    mean_mode = ['columbus', 'ford', 'gas', 'clean', 'automatic', 'sedan', 'ca', 'Mainstream', 'white', 'good', '6 cylinders', '4wd', 2013, 85548.0, 39.1501, -88.4326, 8.0]
    counter = 0
    new_array = []
    for value, dimention in zip(array, dimentions):
        if pd.isna(value) or value is None:
            value = mean_mode[counter]
        
        if counter < 12:
            if value not in list(encodings[dimention].index):
                value = mean_mode[counter]
            value = encodings[dimention][value]
        new_array.append(value)
        counter += 1
    return new_array


def predict(error = 0.1):
    prediction = model.predict([applyEncodings(generateCols(getUserInput()), encodings)])[0]
    lower = prediction - prediction*error
    upper = prediction + prediction*error
    
    print('\n\n' + '='*100)
    print('='*100)
    print(f'\n\nEl mejor precio para el coche especificado se encuentra entre {lower:2f} y {upper:2f} $USD.\n\n')
    print('='*100)
    print('='*100)