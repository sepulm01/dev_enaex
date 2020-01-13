import sqlite3
from sqlite3 import Error
import csv
import numpy as np


class basedatos:

	# Crea tabla registros
	def create(conn):
	    try:
	        # reg_id integer PRIMARY KEY,
	        conn.execute("""CREATE TABLE registros (
	        obj_id integer NOT NULL,
	        clase integer NOT NULL,
	        x1 integer NOT NULL,
	        y1 integer NOT NULL,
	        x2 integer NOT NULL,
	        y2 integer NOT NULL,
	        tiempo text NOT NULL,
	        phi REAL  NOT NULL, 
	        mag REAL  NOT NULL,
	        cam integer  NOT NULL,
	        frm_num integer  NOT NULL
	        );""")
	        print('Crea tabla registros')
	    except Error as e:
	        print(e)
	        pass
	# Crea tabla contador
	def create_contador(conn):
	    try:
	        # reg_id integer PRIMARY KEY,
	        conn.execute("""CREATE TABLE contador (
	        id_cont INTEGER PRIMARY KEY AUTOINCREMENT,
	        camara integer NOT NULL,
	        area integer NOT NULL,
	        obj_id integer NOT NULL,
	        clase integer NOT NULL,
	        fecha text NOT NULL,
	        hora text  NOT NULL,
	        num integer NOT NULL,
	        direc integer NOT NULL
	        );""")
	        print('Crea tabla contador')
	    except Error as e:
	        print(e)
	        pass

	# Crea tabla contador2
	def create_contador2(conn):
	    try:
	        # reg_id integer PRIMARY KEY,
	        conn.execute("""CREATE TABLE contador2 (
	        id_cont INTEGER PRIMARY KEY AUTOINCREMENT,
	        camara integer NOT NULL,
	        area integer NOT NULL,
	        obj_id integer NOT NULL,
	        clase integer NOT NULL,
	        fecha text NOT NULL,
	        hora text  NOT NULL,
	        num integer NOT NULL,
	        direc integer NOT NULL,
	        tipo text NOT NULL,
	        junction  text NOT NULL,
	        arco text NOT NULL,
	        sentido text NOT NULL,
	        viraje text NOT NULL
	        );""")
	        print('Crea tabla contador2')
	    except Error as e:
	        print(e)
	        pass


	def create_connection():
		try:
			#conn = sqlite3.connect('sql/pythonsqlite.db')
			conn = sqlite3.connect(':memory:')
			print('SQLite ver',sqlite3.version)
		except Error as e:
			print(e)
		#finally:
		#	conn.close()
		#Crea TABLAS
		basedatos.create(conn)
		basedatos.create_contador(conn)
		basedatos.create_contador2(conn)
		return conn	
	    

	def inserta(conn, objeto, claseid,x1,y1,x2,y2,tiempo, phi, mag, cam, frm_num ):
	    try:
	        conn.execute("""INSERT INTO registros (
	        obj_id,
	        clase ,
	        x1,
	        y1,
	        x2,
	        y2,
	        tiempo,
	        phi,
	        mag,
	        cam,
	        frm_num)
	        VALUES
	        (?, ?,?,?,?,?,?,?,?,?,?)
	        ;""",(objeto, claseid,x1,y1,x2,y2,tiempo, phi, mag, cam, frm_num))
	        #print('obj insertado frame', objeto, claseid,x1,y1,x2,y2,frame)
	    except Error as e:
	        print('error en inserta registros:',e)
	        pass
	    
	def inserta_contador(conn, camara,area,obj_id,clase,fecha,hora,num,direc):
	    try:
	        conn.execute("""INSERT INTO contador (camara,area,obj_id,clase,fecha,hora,num,direc)
	        VALUES
	        (?,?,?,?,?,?,?,?)
	        ;""",(camara,area,obj_id,clase,fecha,hora,num,direc))
	        #print('obj insertado frame', objeto, claseid,x1,y1,x2,y2,frame)
	    except Error as e:
	        print('error en inserta Contador:',e)
	        pass

	#Tabla modificada con la nomenclatura de los detectores
	def inserta_contador2(conn, camara,area,obj_id,clase,fecha,hora,num,direc,tipo,junction,arco,sentido,viraje):
	    try:
	        conn.execute("""INSERT INTO contador2 (camara,area,obj_id,clase,fecha,hora,num,direc,tipo,junction,arco,sentido,viraje )
	        VALUES
	        (?,?,?,?,?,?,?,?,?,?,?,?,?)
	        ;""",(camara,area,obj_id,clase,fecha,hora,num,direc,tipo,junction,arco,sentido,viraje))
	        #print('obj insertado frame', objeto, claseid,x1,y1,x2,y2,frame)
	    except Error as e:
	        print('error en inserta Contador2:',e)
	        pass


	def consulta(conn):
	    try:
	        cur = conn.cursor()
	        cur.execute("""SELECT * FROM registros""")
	        rows = cur.fetchall()
	        for row in rows:
	            print(row)
	        
	    except Error as e:
	        print('NOOOOOOOOOOO! consulta')
	        print(e)
	        pass

	def consulta_contador(conn,id_cont):
	    try:
	        cur = conn.cursor()
	        cur.execute("""SELECT * FROM contador WHERE id_cont > ?;""", (id_cont,))
	        rows = cur.fetchall()
	        filas = []
	        id_cont = []
	        for row in rows:
	            filas.append((row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]))
	            id_cont = row[0]
	        return filas, id_cont 
	        
	    except Error as e:
	        print('NOOOOOOOOOOO! consulta contador')
	        print(e)
	        pass

	def consulta_contador2(conn,id_cont):
	    try:
	        cur = conn.cursor()
	        cur.execute("""SELECT * FROM contador2 WHERE id_cont > ?;""", (id_cont,))
	        rows = cur.fetchall()
	        filas = []
	        id_cont = []
	        for row in rows:
	            filas.append((row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13]))
	            id_cont = row[0]
	        return filas, id_cont 
	        
	    except Error as e:
	        print('NOOOOOOOOOOO! consulta contador2')
	        print(e)
	        pass	        

	def consulta_obj(conn,id):
	    try:
	        cur = conn.cursor()
	        cur.execute("SELECT x1,y1,x2,y2 FROM registros WHERE obj_id=? Order By tiempo DESC LIMIT 4;", (id,))
	        rows = cur.fetchall()
	        pts = []
	        for row in rows:
	            x, y = basedatos.centro(row[0],row[1],row[2],row[3])
	            pts.append([x,y])
	        ptsout = np.array([pts], np.int32)
	        ptsout = ptsout.reshape((-1,1,2))
	        return ptsout
	        
	    except Error as e:
	        print('NOOOOOOOOOOO! consulta')
	        print(e)
	        pass

	def consulta_obj_clase(conn,id):
	    try:
	        cur = conn.cursor()
	        cur.execute("SELECT x1,y1,x2,y2,clase,mag FROM registros WHERE mag > 3 and obj_id=?;", (id,))
	        rows = cur.fetchall()
	        pts = []
	        clase =[]
	        for row in rows:
	            x, y = basedatos.centro(row[0],row[1],row[2],row[3])
	            pts.append([x,y])
	            clase = row[4]
	        ptsout = np.array([pts], np.int32)
	        ptsout = ptsout.reshape((-1,1,2))
	        return ptsout, clase
	        
	    except Error as e:
	        print('NOOOOOOOOOOO! consulta')
	        print(e)
	        pass

	def cons_obj_dir(conn,id):
	    try:
	        cur = conn.cursor()
	        cur.execute("SELECT x1,y1,x2,y2,tiempo,phi,mag,clase,frm_num FROM registros WHERE mag > 3 and obj_id=?;", (id,))
	        rows = cur.fetchall()
	        pts = []
	        tmp = []
	        phi = []
	        mag =[]
	        clase =[]
	        frm_num = []
	        for row in rows:
	            x, y = basedatos.centro(row[0],row[1],row[2],row[3])
	            pts.append([x,y])
	            tmp.append(row[4])
	            phi.append([row[5]])
	            mag.append(row[6])
	            clase = row[7]
	            frm_num.append(row[8])
	        ptsout = np.array([pts], np.int32)
	        ptsout = ptsout.reshape((-1,1,2))
	        return ptsout, phi, mag, clase, frm_num
	        
	    except Error as e:
	        print('NOOOOOOOOOOO! cons_obj_dir(conn,id)')
	        print(e)
	        pass

	def cons_Kmeans(conn):
	    try:
	        cur = conn.cursor()
	        #cur.execute("SELECT x1,y1,x2,y2,phi,obj_id FROM registros WHERE mag > 0") # antes era 6
	        cur.execute("SELECT x1,y1,x2,y2,phi,obj_id FROM registros ")
	        rows = cur.fetchall()
	        pts = []
	        phi = []
	        xypi = []

	        for row in rows:
	            x, y = basedatos.centro(row[0],row[1],row[2],row[3])
	            pi = row[4]
	            pts.append([x,y])
	            phi.append([pi])
	            xypi.append([x,y,pi,row[5]])
	        ptsout = np.array([pts], np.int32)
	        ptsout = ptsout.reshape((-1,2))
	        xypi_out = np.array([xypi], np.int32)
	        xypi_out = xypi_out.reshape((-1,4))
	        return ptsout, phi, xypi_out
	        
	    except Error as e:
	        print('NOOOOOOOOOOO! consulta Kmeans')
	        print(e)
	        pass
	def cons_ori(conn):
	    try:
	        cur = conn.cursor()
	        #cur.execute("SELECT x1,y1,x2,y2,phi,obj_id FROM registros WHERE mag > 3  GROUP BY obj_id LIMIT 10")
	        cur.execute("SELECT x1,y1,x2,y2,phi,obj_id FROM registros  WHERE mag > 3")
	        rows = cur.fetchall()
	        pts = []
	        phi = []
	        id_=[]
	        for row in rows:
	            x, y = basedatos.centro(row[0],row[1],row[2],row[3])
	            pi = row[4]
	            pts.append([x,y])
	            phi.append([pi])
	            id_.append([row[5]])
	        ptsout = np.array([pts], np.int32)
	        ptsout = ptsout.reshape((-1,2))
	        return ptsout, phi, id_
	        
	    except Error as e:
	        print('NOOOOOOOOOOO! consulta Kmeans')
	        print(e)
	        pass
	def export_csv(conn, base_csv):
	    #with open('base.csv', 'w', newline='') as myfile:
	     #   wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_NONE)
	   with open(base_csv, 'w+') as write_file:
	        cursor = conn.cursor()
	        cabecera = 'objeto,claseid,x1,y1,x2,y2,tiempo,phi,mag,cam' +'\n'
	        write_file.write(cabecera)
	        for row in cursor.execute('SELECT * FROM registros'):
	        # use the cursor as an iterable
	            lineadb = str( row )
	            #print('antes',lineadb)
	            lineadb = lineadb[1:len(lineadb)-1]+'\n'
	            #print('despu',lineadb)
	            write_file.write(lineadb)
	   print("export tabla registros hecho!")

	def export_cont_csv(conn, base_csv):
	    #with open('base.csv', 'w', newline='') as myfile:
	     #   wr = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_NONE)
	   with open(base_csv, 'w+') as write_file:
	        cursor = conn.cursor()
	        cabecera = 'id_cont,camara,area,obj_id,clase,fecha,hora,num' +'\n'
	        write_file.write(cabecera)
	        for row in cursor.execute('SELECT * FROM contador'):
	        # use the cursor as an iterable
	            lineadb = str( row )
	            #print('antes',lineadb)
	            lineadb = lineadb[1:len(lineadb)-1]+'\n'
	            #print('despu',lineadb)
	            write_file.write(lineadb)
	   print("export tabla Contador hecho!")

	def centro(x1,y1,x2,y2):
	    cx = int(x1+(x2-x1)/2)
	    cy = int(y1+(y2-y1)/2)
	    return cx,cy

	def borra(conn):
	    try:
	        cur = conn.cursor()
	        cur.execute("""DELETE FROM registros""")
	        cur.execute("""DELETE FROM contador""")
	        conn.isolation_level = None
	        conn.execute("vacuum;").close()

	        
	    except Error as e:
	        print('NOOOOOOOOOOO! borra')
	        print(e)
	        pass
