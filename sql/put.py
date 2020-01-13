import psycopg2
import json
import matplotlib.path as mpltPath
import numpy as np

class bdremota:
	def __init__(self):
		self.user = 'postgres'
		self.password = 'mysecretpassword'
		self.host = '172.17.0.2'
		self.port = '5432'
		self.database = 'streetflow'



	def alarma_pict(self,archivo):
		'''
		Inserta registros en tabla images base postgres.
		'''
		try:
			drawing = open(archivo, 'rb').read()
			connection = psycopg2.connect(
				user = self.user,
				password = self.password,
				host = self.host,
				port = self.port,
				database = self.database
				)
			cursor = connection.cursor()
			postgres_insert_query = """ INSERT INTO images (  blob )
										 VALUES (%s)"""
			#print(records_to_insert,"records_to_insert")
			cursor.execute(postgres_insert_query, (psycopg2.Binary(drawing),))
			connection.commit()
			count = cursor.rowcount
			print (count, "imagen exitosamente insertados en la tabla 'images'")
		except (Exception, psycopg2.Error) as error :
			#if(connection):
			print("Falla insertando imagen en tabla 'images' {}".format(error))
		finally:
			#closing database connection.
			if(connection):
				cursor.close()
				connection.close()
				print("PostgreSQL connection is closed")


	def get(self):
		'''
		Busca los parametros basicos para inicializar el flujo de video (base postgres).
		'''
		try:
			connection = psycopg2.connect(user = "postgres",
				password = "mysecretpassword",
				host = "172.17.0.2",
				port = "5432",
				database = "streetflow")
			cursor = connection.cursor()
			postgreSQL_select_Query = """ select id, blob from images where id=3 """
			cursor.execute(postgreSQL_select_Query)
			blob = cursor.fetchone()
			#open('new.jpg', 'wb').write(blob[1])

		except (Exception, psycopg2.Error) as error :
			if(connection):
				print("Failed to insert record into mobile table", error)
		finally:
			#closing database connection.
			if(connection):
				cursor.close()
				connection.close()
				print("PostgreSQL connection is closed")


		return blob[1]

def main():

	put=bdremota()
	put.alarma_pict('../output/alarma.0.jpg')
	#frame=put.get()

	#open('new1.jpg', 'wb').write(frame)

if __name__ == '__main__':
    main()