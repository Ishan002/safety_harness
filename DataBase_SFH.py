import datetime
import random
import time
import mysql.connector as sql

class dbconnect():
    def __init__(self, hname, usr, pwd, datab, tablenm):
        self.db = sql.connect(
            host=hname,
            user=usr,
            password=pwd,
            database=datab,
            # auth_plugin='mysql_native_password'
        )
        if self.db.is_connected() == False:
            print("not connected")
        if self.db.is_connected() == True:
            print(" connected")
        self.curs = self.db.cursor()
        self.tablename = tablenm
      

    def add_dbdata(self, data):
        timestmp = str(datetime.datetime.now().strftime("%H:%M:%S"))
        print(data[0],data[1],data[2],data[3])
        datenm = str(datetime.datetime.today().date())
        try:
            self.curs.execute(f"INSERT INTO harness_monitoring(Date,Time,Camname,Track_ID,Class,Duration) VALUES (%s, %s, %s, %s, %s, %s)",(datenm, timestmp, data[0], data[1], data[2],data[3]))
            self.db.commit()
            print("Added data to database!")
        except sql.Error as e:
            print("Data push failed:", e)

            


