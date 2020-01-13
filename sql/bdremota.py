#!/usr/bin/env python

'''
Recdata SPA.
Feb - 2019
Remote MySQL  sql154.main-hosting.eu.
dabase u308327591_uoct
username u308327591_sepul

'''
#import mysql.connector
#from mysql.connector import Error

import psycopg2 #Conector Postgres

class bdremota:
	def __init__(self):
		self.user = 'postgres'
		self.password = 'mysecretpassword'
		self.host = '172.17.0.2'
		self.port = '5432'
		self.database = 'streetflow'

	def ins_5min(self,records_to_insert):
		try:
		    connection = mysql.connector.connect(host='localhost',
		                             database='streetflow',
		                             user='root',
		                             password='123456')
		    print("tratando", connection.is_connected())
		    if connection.is_connected():
		    	#print(records_to_insert,"records_to_insert")
		    	#records_to_insert = [(0, 0, 2, 0, '2019-03-27', '21:23:44', 1) ,(0,0,14012,0,'2019-03-28','15:03:30',1),(0,0,999,0,'2019-03-28','15:03:30',1)]
		    	sql_insert_query = """ INSERT INTO `contador` (`camara`, `area`, `obj_id`, `clase`, `fecha`, `hora`, `num`, `direc` ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
		    	cursor = connection.cursor()
		    	result  = cursor.executemany(sql_insert_query, records_to_insert)
		    	connection.commit()
		    print (cursor.rowcount, "Registros exitosamente insertados en la tabla 'contador'")


		except mysql.connector.Error as error :
		    print("Falla insertando registro en tabla 'contador' {}".format(error))
		finally:
		    #closing database connection.
		    if(connection.is_connected()):
		        cursor.close()
		        connection.close()
		        print("la coneccion se cerro")

	def upd_pict(self,records_to_insert):
		try:
		    # connection = mysql.connector.connect(host='sql154.main-hosting.eu',
		    #                          database='u308327591_uoct',
		    #                          user='u308327591_sepul',
		    #                          password='rguax910')
		    connection = mysql.connector.connect(host='localhost',
		                             database='streetflow',
		                             user='root',
		                             password='123456')


		    print("tratando", connection.is_connected())
		    if connection.is_connected():
		    	#print(records_to_insert,"records_to_insert")
		    	#records_to_insert = [(0, 0, 2, 0, '2019-03-27', '21:23:44', 1) ,(0,0,14012,0,'2019-03-28','15:03:30',1),(0,0,999,0,'2019-03-28','15:03:30',1)]
		    	sql_insert_query = """ INSERT INTO `imagenes` (`camara`, `area`, `obj_id`, `clase`, `fecha`, `hora`, `num`, `direc`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
		    	cursor = connection.cursor()
		    	result  = cursor.executemany(sql_insert_query, records_to_insert)
		    	connection.commit()
		    print (cursor.rowcount, "Registros exitosamente insertados en la tabla 'contador'")


		except mysql.connector.Error as error :
		    print("Falla insertando registro en tabla 'contador' {}".format(error))
		finally:
		    #closing database connection.
		    if(connection.is_connected()):
		        cursor.close()
		        connection.close()
		        print("la coneccion se cerro")

	def ins_postgres(self,records_to_insert):
		'''
		Inserta registros en tabla contador base postgres.
		'''
		try:
			connection = psycopg2.connect(
				user = self.user,
				password = self.password,
				host = self.host,
				port = self.port,
				database = self.database
				)
			cursor = connection.cursor()
			postgres_insert_query = """ INSERT INTO contador (camara, area, obj_id, clase, fecha, hora, num, direc ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
			cursor.executemany(postgres_insert_query, records_to_insert)
			connection.commit()
			count = cursor.rowcount
			print (count, "registros exitosamente insertados en la tabla 'contador'")
		except (Exception, psycopg2.Error) as error :
			#if(connection):
			print("Falla insertando registro en tabla 'contador' {}".format(error))
		finally:
			#closing database connection.
			if(connection):
				cursor.close()
				connection.close()
				print("PostgreSQL connection is closed")

	def ins_postgres2(self,records_to_insert):
		'''
		Inserta registros en tabla contador base postgres.
		'''
		try:
			connection = psycopg2.connect(
				user = self.user,
				password = self.password,
				host = self.host,
				port = self.port,
				database = self.database
				)
			cursor = connection.cursor()
			postgres_insert_query = """ INSERT INTO contador2 (camara, area, obj_id, clase, fecha, hora, num, direc, tipo, junction, arco, sentido, viraje ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
			#print(records_to_insert,"records_to_insert")
			cursor.executemany(postgres_insert_query, records_to_insert)
			connection.commit()
			count = cursor.rowcount
			print (count, "registros exitosamente insertados en la tabla 'contador2'")
		except (Exception, psycopg2.Error) as error :
			#if(connection):
			print("Falla insertando registro en tabla 'contador2' {}".format(error))
		finally:
			#closing database connection.
			if(connection):
				cursor.close()
				connection.close()
				print("PostgreSQL connection is closed")	

	def alarma_post(self,records_to_insert, foto_file):
		'''
		Inserta registros en tabla alarma base postgres.
		'''
		try:
			drawing = open(foto_file, 'rb').read()
			records_to_insert.append(psycopg2.Binary(drawing))
			connection = psycopg2.connect(
				user = self.user,
				password = self.password,
				host = self.host,
				port = self.port,
				database = self.database
				)
			cursor = connection.cursor()
			postgres_insert_query = """ INSERT INTO alarmas (camara, area, fecha, hora, tipo, junction, arco, sentido, viraje, tipo_alarm, seg_alarm, perd_alarm, foto )
										 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, %s)"""
			#print(records_to_insert,"records_to_insert")
			cursor.execute(postgres_insert_query, records_to_insert)
			connection.commit()
			count = cursor.rowcount
			print (count, "registros exitosamente insertados en la tabla 'alarmas'")
		except (Exception, psycopg2.Error) as error :
			#if(connection):
			print("Falla insertando registro en tabla 'alarmas' {}".format(error))
		finally:
			#closing database connection.
			if(connection):
				cursor.close()
				connection.close()
				print("PostgreSQL connection is closed")

	def alarma_pict(self,records_to_insert):
		'''
		Inserta registros en tabla images base postgres.
		'''
		try:
			drawing = open('output/alarma.0.jpg', 'rb').read()
			connection = psycopg2.connect(
				user = self.user,
				password = self.password,
				host = self.host,
				port = self.port,
				database = self.database
				)
			cursor = connection.cursor()
			postgres_insert_query = """ INSERT INTO images ( blob, tiempo=NOW() )
										 VALUES (%s)"""
			#print(records_to_insert,"records_to_insert")
			cursor.executemany(postgres_insert_query, psycopg2.Binary(drawing))
			connection.commit()
			count = cursor.rowcount
			print (count, "imagen exitosamente insertados en la tabla 'alarmas_pict'")
		except (Exception, psycopg2.Error) as error :
			#if(connection):
			print("Falla insertando imagen en tabla 'alarmas_pict' {}".format(error))
		finally:
			#closing database connection.
			if(connection):
				cursor.close()
				connection.close()
				print("PostgreSQL connection is closed")

	def get_ini(self,flag, id_):
		'''
		Busca los parametros basicos para inicializar el flujo de video (base postgres).
		'''
		if flag == 0:
			try:
				connection = psycopg2.connect(
					user = self.user,
					password = self.password,
					host = self.host,
					port = self.port,
					database = self.database )
				cursor = connection.cursor()
				postgreSQL_select_Query = """ select id, estado, details from flujos_videos where estado=0 """
				cursor.execute(postgreSQL_select_Query)
				param_records = cursor.fetchall()

			except (Exception, psycopg2.Error) as error :
				connection=False
				print("Fallo obtener datos de inicializacion desde base de datos.\n", error)
				param_records = []
			finally:
				#closing database connection.
				if(connection):
					cursor.close()
					connection.close()
					print("PostgreSQL connection is closed")
			if len(param_records)>0 and connection:
				id_ = param_records[0][0]
				param_records = param_records[0][2]
				
			return param_records,id_

		elif flag == 1:
			try:
	
				connection = psycopg2.connect(user = self.user,
					password = self.password,
					host = self.host,
					port = self.port,
					database = self.database)
				cursor = connection.cursor()
				postgres_insert_query = """ update flujos_videos set estado = 1, flujo_tstmp =NOW() where id=%s; """
				cursor.execute(postgres_insert_query, (id_,))
				connection.commit()
				count = cursor.rowcount
				print (count, "Tabla flujos video actualizada id:%d a estado tomado(1)."%id_)
			except (Exception, psycopg2.Error) as error :
				#if(connection):
				print("Fallo en update tabla flujos video", error)
				print(str(id_), id_, type(id_))
			finally:
				#closing database connection.
				if(connection):
					cursor.close()
					connection.close()
					print("PostgreSQL connection is closed.")


		elif flag == 2:
			try:
				connection = psycopg2.connect(user = self.user,
					password = self.password,
					host = self.host,
					port = self.port,
					database = self.database)
				cursor = connection.cursor()
				postgres_insert_query = """ update flujos_videos set estado = 2, flujo_tstmp =NOW() where id=%s """
				cursor.execute(postgres_insert_query, (id_,))
				connection.commit()
				count = cursor.rowcount
				print (count, " Update estado(2) tabla flujos de video, ie funcionando. ")
			except (Exception, psycopg2.Error) as error :
				#if(connection):
				print("Fallo Update estado(2) tabla flujos de video.", error)
			finally:
				#closing database connection.
				if(connection):
					cursor.close()
					connection.close()
					print("PostgreSQL connection is closed.")

		elif flag == 3:
			try:
				connection = psycopg2.connect(user = self.user,
					password = self.password,
					host = self.host,
					port = self.port,
					database = self.database)
				cursor = connection.cursor()
				postgres_insert_query = """ update flujos_videos set estado = 3, flujo_tstmp =NOW() where id=%s """
				cursor.execute(postgres_insert_query, (id_,))
				connection.commit()
				count = cursor.rowcount
				print (count, "Update estado(3) tabla flujos de video, ie fuente sin señal (ret). ")
			except (Exception, psycopg2.Error) as error :
				#if(connection):
				print("Fallo Update estado(3) tabla flujos de video.", error)
			finally:
				#closing database connection.
				if(connection):
					cursor.close()
					connection.close()
					print("PostgreSQL connection is closed.")

		elif flag == 4:
			try:
				connection = psycopg2.connect(user = self.user,
					password = self.password,
					host = self.host,
					port = self.port,
					database = self.database)
				cursor = connection.cursor()
				postgres_insert_query = """ update flujos_videos set estado = 0, flujo_tstmp =NOW() where id=%s """
				cursor.execute(postgres_insert_query, (id_,))
				connection.commit()
				count = cursor.rowcount
				print (count, "Update estado a 0 (disponible) tabla flujos de video, progama terminado (break).")
			except (Exception, psycopg2.Error) as error :
				#if(connection):
				print("Fallo Update estado(4) tabla flujos de video.", error)
			finally:
				#closing database connection.
				if(connection):
					cursor.close()
					connection.close()
					print("PostgreSQL connection is closed.")

		elif flag == 5:
			try:
				connection = psycopg2.connect(user = self.user,
					password = self.password,
					host = self.host,
					port = self.port,
					database = self.database)
				cursor = connection.cursor()
				postgres_insert_query = """ update flujos_videos set estado = 5, flujo_tstmp =NOW() where id=%s """
				cursor.execute(postgres_insert_query, (id_,))
				connection.commit()
				count = cursor.rowcount
				print (count, "Update estado a 5 tabla flujos de video, No hay flujo de video (ret==0).")
			except (Exception, psycopg2.Error) as error :
				#if(connection):
				print("Fallo Update estado(5) tabla flujos de video.", error)
			finally:
				#closing database connection.
				if(connection):
					cursor.close()
					connection.close()
					print("PostgreSQL connection is closed.")

		elif flag == 10:
			'''
			Revisa el estado de los flujos
			'''
			try:
				connection = psycopg2.connect(
					user = self.user,
					password = self.password,
					host = self.host,
					port = self.port,
					database = self.database )
				cursor = connection.cursor()
				postgreSQL_select_Query = """ select id, details ->>'junction' AS junction, estado, flujo_tstmp from flujos_videos """
				cursor.execute(postgreSQL_select_Query)
				param_records = cursor.fetchall()

			except (Exception, psycopg2.Error) as error :
				connection=False
				print("Fallo obtener datos de flujos desde base de datos.\n", error)
				param_records = []
			finally:
				#closing database connection.
				if(connection):
					cursor.close()
					connection.close()
					print("PostgreSQL connection is closed")
				
			return param_records


