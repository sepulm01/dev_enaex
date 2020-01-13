import psycopg2


try:
    connection = psycopg2.connect(user = "postgres",
                                  password = "mysecretpassword",
                                  host = "172.17.0.3",
                                  port = "5432",
                                  database = "streetflow")
    cursor = connection.cursor()

    postgres_insert_query = """ INSERT INTO contador (camara, area, obj_id, clase, fecha, hora, num, direc ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
    record_to_insert = (4, 0, 1, 0, '2019-05-24', '15:48:19', 1, 1)
    cursor.executemany(postgres_insert_query, record_to_insert)
    connection.commit()
    count = cursor.rowcount
    print (count, "Record inserted successfully into mobile table")

except (Exception, psycopg2.Error) as error :
    if(connection):
        print("Failed to insert record into mobile table", error)
finally:
    #closing database connection.
    if(connection):
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")