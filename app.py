import pandas as pd 
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score


pickle_in = open('xgb.pkl', 'rb')
model = pickle.load(pickle_in)

dummies = pd.read_csv('dummies.csv')
del dummies['Unnamed: 0']


st.title("Hotel Booking Demand")

st.sidebar.write("In tourism and travel related industries, most of the research on Revenue Management demand forecasting and prediction problems employ data"
"from the aviation industry, in the format known as the Passenger Name Record (PNR). This is a format developed by the aviation industry. However, the "
"remaining tourism and travel industries like hospitality, cruising, theme parks, etc., have different requirements and particularities that cannot be fully "
"explored without industry׳s specific data. Hence, two hotel datasets with demand data are shared to help in overcoming this limitation.")


#VARIABLES-----------------------------------------------------------------
number_of_cancellation = st.number_input("Insert the number of previous cancellations:")
adr = st.number_input("Insert the price (USD):")
adults = st.number_input("Insert number of adults:")
agent = st.number_input("Insert the ID of the travel agency:")
arrival_date_week_number = st.number_input("Insert the week number:")
assigned_room_type = st.selectbox("Insert the room type:", ['C', 'A', 'D', 'E', 'G', 'F', 'I', 'B', 'H', 'L', 'K'])
babies = st.number_input("Insert number of babies (0 - 5):")
booking_changes = st.number_input("Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation:")
children = st.number_input("Insert number of children (5 - 18):")
country = st.selectbox("Insert Country:", ['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'OMN',
                                            'ARG', 'DEU', 'POL', 'BEL', 'BRA', 'ITA', 'CHE', 'CN', 'GRC',
                                            'NLD', 'DNK', 'RUS', 'SWE', 'AUS', 'EST', 'CZE', 'FIN', 'AUT',
                                            'HUN', 'ISR', 'MOZ', 'BWA', 'NZL', 'LUX', 'IDN', 'SVN', 'ALB',
                                            'MAR', 'CHN', 'HRV', 'AGO', 'BGR', 'IND', 'DZA', 'MEX', 'TUN',
                                            'COL', 'KAZ', 'LVA', 'STP', 'UKR', 'VEN', 'TWN', 'IRN', 'SMR',
                                            'KOR', 'TUR', 'BLR', 'JPN', 'PRI', 'SRB', 'LTU', 'CPV', 'AZE',
                                            'LBN', 'CRI', 'CHL', 'THA', 'SVK', 'EGY', 'CMR', 'LIE', 'MYS',
                                            'SAU', 'ZAF', 'MKD', 'MMR', 'DOM', 'IRQ', 'SGP', 'CYM', 'ZMB',
                                            'PAN', 'ZWE', 'SEN', 'NGA', 'GIB', 'ARM', 'PER', 'KNA', 'JOR',
                                            'KWT', 'LKA', 'GEO', 'TMP', 'ETH', 'MUS', 'ECU', 'PHL', 'CUB',
                                            'ARE', 'BFA', 'AND', 'CYP', 'KEN', 'BIH', 'COM', 'SUR', 'JAM',
                                            'HND', 'MCO', 'GNB', 'RWA', 'LBY', 'PAK', 'UGA', 'TZA', 'CIV',
                                            'SYR', 'QAT', 'KHM', 'HKG', 'BGD', 'MLI', 'ISL', 'UZB', 'BHR',
                                            'URY', 'NAM', 'BOL', 'IMN', 'BDI', 'TJK', 'MDV', 'MLT', 'NIC',
                                            'SYC', 'PRY', 'BRB', 'ABW', 'GGY', 'VNM', 'AIA', 'SLV', 'PLW',
                                            'BEN', 'MAC', 'DMA', 'VGB', 'JEY', 'GAB', 'CAF', 'PYF', 'LCA',
                                            'GUY', 'ATA', 'GHA', 'MWI', 'MNE', 'GLP', 'GTM', 'MDG', 'ASM',
                                            'TGO', 'NPL', 'MRT', 'BHS', 'UMI', 'NCL', 'FJI', 'KIR', 'SDN',
                                            'MYT', 'ATF', 'DJI', 'SLE', 'FRO', 'LAO'])
days_in_waiting_list =  st.number_input("Number of days the booking was in the waiting list before it was confirmed to the customer:")
hotel = st.selectbox("Insert the hotel type:", ['Resort Hotel', 'City Hotel'])
is_repeated_guest = st.number_input("Value indicating if the booking name was from a repeated guest (1) or not (0):")
lead_time = st.number_input("Number of days that elapsed between the entering date of the booking into the PMS and the arrival date")
meal = st.selectbox("Insert the treatment type:", ['BB', 'FB', 'HB', 'SC', 'Undefined'])
previous_bookings_not_canceled = st.number_input("Number of previous bookings not cancelled by the customer prior to the current booking:")
previous_cancellations = st.number_input("Number of previous bookings that were cancelled by the customer prior to the current booking:")
required_car_parking_spaces = st.number_input("Number of car parking spaces required by the customer:")
reserved_room_type = st.selectbox("Insert the room type:", ['C', 'A', 'D', 'E', 'G', 'F', 'I', 'B', 'H', 'L'])
stays_in_week_nights = st.number_input("Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel:")
stays_in_weekend_nights = st.number_input("	Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel:")
total_of_special_requests = st.number_input("Number of special requests made by the customer (e.g. twin bed or high floor):")
family = adults + babies + children


col = {'number_of_cancellation' : number_of_cancellation,
        'adr' : adr,
        'adults' : adults,
        'agent' : agent,
        'arrival_date_week_number' :arrival_date_week_number, 
        'assigned_room_type' : assigned_room_type,
        'babies': babies, 
        'booking_changes' : booking_changes, 
        'children' : children, 
        'country' : country, 
        'days_in_waiting_list' : days_in_waiting_list, 
        'hotel' : hotel, 
        'is_repeated_guest' : is_repeated_guest, 
        'lead_time' : lead_time, 
        'meal' : meal, 
        'previous_bookings_not_canceled' : previous_bookings_not_canceled, 
        'previous_cancellations' : previous_cancellations,
        'required_car_parking_spaces' : required_car_parking_spaces, 
        'reserved_room_type' : reserved_room_type, 
        'stays_in_week_nights' : stays_in_week_nights, 
        'stays_in_weekend_nights' : stays_in_weekend_nights, 
        'total_of_special_requests' : total_of_special_requests, 
        'family' : family}


df = pd.DataFrame(col, index = [0])

if st.button('Predict'):

    #Features Engineering
    df = pd.get_dummies(df)
    missing_cols = set(dummies.columns) - set(df.columns)
    for c in missing_cols:
        df[c] = 0

    df = df[dummies.columns]


    sc_X = StandardScaler()
    df2 = pd.DataFrame(sc_X.fit_transform(df))
    df2.columns = df.columns.values
    df2.index = df.index.values
    df = df2

    pred = model.predict(df)
    probs = model.predict_proba(df) 
    probs = probs[:, 0] 
    if pred == 0:
        st.write('The reservation should not be canceled')
    else:
        st.write('the reservation should be canceled')

    
