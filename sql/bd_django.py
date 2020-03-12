import sqlite3
from sqlite3 import Error
import csv
import numpy as np


class basedatos:

	def create_connection():
		try:
			conn = sqlite3.connect('mysite/db.sqlite3')
			#conn = sqlite3.connect(':memory:')
			print('SQLite ver',sqlite3.version)
		except Error as e:
			print(e)
		#finally:
		#	conn.close()
		return conn	

	def cam_disp(conn):
		try:
			cur = conn.cursor()
			cur.execute("""SELECT * FROM camaras_camara WHERE estado=0""")
			rows = cur.fetchall()
			filas = []
			for row in rows:
				#print(row)
				filas.append([row[0],row[1],row[3],row[4]])
			return filas

		except Error as e:
			print('NOOOOOOOOOOO! consulta')
			print(e)
			pass

	def cam_ocupada(conn, est, pk):
		sql = """UPDATE camaras_camara SET estado = ? WHERE id=?;"""
		try:
			cur = conn.cursor()
			cur.execute(sql,(est,pk,))
			conn.commit()
		except Error as e:
			print('error en update cam_ocupada:',e)
			pass

	def cam_viva(conn, est, pk, img, actualizado):
		sql = """UPDATE camaras_camara SET estado = ?, image = ?, actualizado= ? WHERE id=?;"""
		try:
			cur = conn.cursor()
			cur.execute(sql,(est,img, actualizado, pk,))
			conn.commit()
		except Error as e:
			print('error en update cam_viva:',e)
			pass

	def alarma(conn, camara_id,tiempo ,clase,cantidad,video ):
	    try:
	        conn.execute("""INSERT INTO camaras_alarmas (
	        camara_id,
	        tiempo ,
	        clase,
	        cantidad,
	        video
	        )
	        VALUES
	        (?,?,?,?,?)
	        ;""",(camara_id,tiempo ,clase,cantidad,video))
	        conn.commit()
	        #print('obj insertado frame', objeto, claseid,x1,y1,x2,y2,frame)
	    except Error as e:
	        print('error en inserta alarmas:',e)
	        pass