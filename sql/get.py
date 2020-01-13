import psycopg2
import json
import matplotlib.path as mpltPath
import numpy as np

def get_ini():
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
		postgreSQL_select_Query = """ select id, estado, details from flujos_video where estado=0 """
		cursor.execute(postgreSQL_select_Query)
		param_records = cursor.fetchall()

	except (Exception, psycopg2.Error) as error :
		if(connection):
			print("Failed to insert record into mobile table", error)
	finally:
		#closing database connection.
		if(connection):
			cursor.close()
			connection.close()
			print("PostgreSQL connection is closed")
	if len(param_records)>0:
		id_ = param_records[0][0]
		param_records = param_records[0][2]

	return param_records, id_

def mapa_areas_bd(det_):
	'''Carga areas desde base datos, entrega un objeto tipo mpltPath
	y una lista con N np array (1x2) con los puntos x,y del perimetro del area.
	Donde el N es el id del area o detector.
	'''
	areas = []
	lista=[]
	for z in det_:
		par = []
		for i in range(0,len(det_[z]),2):
			par.append([det_[z][i],det_[z][i+1]])
		lista.append(par)
	for w in range(0,len(lista)):
		asd = np.array(lista[w], np.int32)
		areas.append(asd)
	mapa = []
	dibujo = []
	for z in range(0,len(areas)): 
		mapa.append(mpltPath.Path(areas[z]))
		dibujo.append(areas[z])
	return mapa, dibujo

parametros, id_ = get_ini()

fuente = parametros['fuente']
clases = parametros['clases']
sensib_min = parametros['sensib_min']
sensib_max = parametros['sensib_max']
bgshow = parametros['bgshow']
gammashow = parametros['gammashow']
b_ground_sub = parametros['b_ground_sub']
area_mask = parametros['area_mask']
contadores = parametros['contadores']
buf_conges = parametros['buf_conges']
gpu_server = parametros['gpu_server']
gpu_port = parametros['gpu_port']
retardo = parametros['retardo']
detec = parametros['detec']

print(type(detec),detec)

mapa_areas_bd(detec)
