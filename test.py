#function to get the current date and time

def get_date_time():
    import datetime
    return datetime.datetime.now()

print(get_date_time())