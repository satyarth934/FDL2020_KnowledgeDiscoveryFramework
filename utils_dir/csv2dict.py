import pandas as pd


# CONVERTING THE CSV GROUND TRUTH TO A DICTIONARY FOR EASY USAGE 

def date_to_nth_day(date, format='%Y%m%d'):
    date = pd.to_datetime(date, format=format)
    new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1


def formatDate(date):
    date_split = date.split()
    f = "".join([date_split[-1], month_dict[date_split[0]], "%02d" % (int(date_split[1].strip(",")))])
    nth_day = date_to_nth_day(f)
    return (date_split[-1] + "%03d" % (nth_day))


def main():
    month_dict = {"January": '01', "February": '02', "March": '03', "April": '04', "May": '05', "June": '06', 
                  "July": '07', "August": '08', "September": '09', "October": '10', "November": '11', "December": '12'}

    # Read CSV file for Ground Truth
    dust_csv = "/home/satyarth934/data/modis_data_products/MODIS_Dust_Events_2010_2020_h16v7.csv"

    dc = pd.read_csv(dust_csv)
    dc["Day"] = [formatDate(d) for d in dc["Day"]]
    dust_gt = dict(zip(dc["Day"], dc["Category"]))
    pickle.dump(dust_gt, file=open(("/home/satyarth934/data/modis_data_products/MODIS_Dust_Events_2010_2020_h16v7.pkl"), mode = 'wb'))

    
if __name__ == "__main__":
    main()