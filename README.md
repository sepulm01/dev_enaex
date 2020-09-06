# dev_enaex


# Volumen NFS

Server:
sudo apt install nfs-kernel-server
nano /etc/exports

incluir:
/var/www/mysite/media   192.168.1.0/24(rw,no_subtree_check,async)

192.168.1.0/24 comparte con toda la red:

sudo exportfs -arv

sudo systemctl enable nfs-kernel-server
sudo systemctl start nfs-kernel-server

Cliente:
  sudo apt install nfs-common
  sudo mkdir -p /mnt/cloudstation/
  sudo mount 192.168.1.51:/var/www/mysite/media /mnt/cloudstation/
  sudo mount 192.168.1.51:/var/www/mysite/media mnt/alarmas/

si sale ok entonces editar el /etc/fstab
nano /etc/fstab

192.168.1.51:/var/www/mysite/media /mnt/cloudstation/ nfs rw,async 0 0

192.168.1.51:/var/www/mysite/media /home/martin/Documents/dev_enaex/mnt/alarmas/ nfs rw,async 0 0

luego desmontar y montar con con:
sudo umount /mnt/cloudstation/
sudo mount -a


# Configuración detector mul rq:

Habilitar archivo conf.file en el mismo direcorio de dector_mul_rq.py
contenido tipo json:

    {
 "website": "http://127.0.0.1:8000"
}


