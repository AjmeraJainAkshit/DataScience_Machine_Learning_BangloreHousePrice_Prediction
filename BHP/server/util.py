import json
import pickle
import numpy as np

#three global variables
__locations = None  # will contains all the locations
__data_columns = None
__model = None



def get_estimated_price(location,total_sqft,bhk,bath): # will return estimated price on the basis of given parameters
    try:
        loc_index = __data_columns.index(location.lower()) # for finding index in python list, i am using .index and location.lower() is
                                                           # used to convert values(which we get) should be in lower case because in 
                                                           # json it is available in lower case format
    except:
        loc_index = -1    

    x = np.zeros(len(__data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2) # we will call predict method from sklearn model which will take x as imput
                                # and it will return estimated price as an output. Here x is 2-d array

def get_location_names(): # will return all location names from column.json starting from "1st block jayanagar" 
    return __locations

def load_saved_artifacts(): # will load the saved artifacts in column.json and banglore_home_price
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
 
    # load data from columns.json
    with open("./artifacts/columns.json",'r') as f: # 'r' stands for read the 'columns.json
        __data_columns = json.load(f)["data_columns"]  # will call data_columns as a key of dictionary which is created 
        __locations = __data_columns[3:]  # slicer will extract the data from index no 3 and put in __location 

    global __model
    # load data from pickel file 
    with open('./artifacts/banglore_home_prices_model.pickle','rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")    
     
if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 2, 2))  # other location 
