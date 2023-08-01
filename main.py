import streamlit as st
import pickle as pkl

from predict import COLUMNS, BRANDS

def main():
    with open('model.pkl', 'br') as file:
        model = pkl.load(file)
        
    with open('encodings.pkl', 'br') as file:
        encodings = pkl.load(file)

    st.header('Car price prediction.')
    st.write(COLUMNS)
    manufacturer = st.selectbox(label = 'manufacturer', options = encodings['manufacturer'].keys())
    fuel = st.selectbox(label = 'fuel type', options = encodings['fuel'].keys())
    type_ = st.selectbox(label = 'fuel type', options = encodings['type'].keys())
    year = st.slider(label = 'year',
                     min_value = 1980,
                     max_value = 2023,
                     value = 2015,
                     step = 1,
                     )
    kms = st.slider(label = 'kilometers',
                    min_value = 0.0,
                    max_value = 1_000_000.0,
                    value = 0.0,
                    step = 0.1,
                    )
    
    cat = [manufacturer, fuel, type_]

    for i, e in enumerate(cat):
        cat[i] = encodings[COLUMNS[i]][e]

    query = cat + [year, kms, -80, 2023-year]

    if st.button(label = 'get price'):
        st.write(model.predict([query]))

if __name__ == '__main__':
    main()