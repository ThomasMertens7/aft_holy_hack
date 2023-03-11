import datetime as dt
import requests
# pip install requests
API_KEY = "43319b8fec5105f0a7a0febe31c7bb42"

#usage:
#print(get_avg_temp_celcius())

def get_location():
    """
    asks the user for a location input
        """
    return input("where do you live? ")
def get_todays_min_max(CITY):
    """
        returns the minimum and maximum temperature predicted for a given city
        """
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
    url = BASE_URL + "appid=" + API_KEY + "&q=" + CITY
    response = requests.get(url).json()
    return response['main']['temp_min'], response['main']['temp_max']

def get_5_day_average(CITY):
    """
        returns the average temperature expected for the next few days
        """
    BASE_URL = "http://api.openweathermap.org/data/2.5/forecast?q="
    url = BASE_URL + CITY + "&appid=" + API_KEY
    response = requests.get(url).json()
    forecasts = response["list"]
    avg = 0
    i = 0
    for forecast in forecasts:
        min_temp = forecast['main']['temp_min']
        max_temp = forecast['main']['temp_max']
        avg += (min_temp+max_temp)/2
        i += 1
    avg = avg/i
    return avg

def kelvin_to_celcius(temp):
    """
        convert degrees celcius to kelvin
        """
    return temp - 273.15

def get_avg_temp_celcius():
    """
        get the average temperature predicted for the next 5 days
        """
    return kelvin_to_celcius(get_5_day_average(get_location()))

