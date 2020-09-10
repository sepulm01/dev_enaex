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
  sudo mount 192.168.1.53:/var/www/mysite/media mnt/alarmas/

si sale ok entonces editar el /etc/fstab
nano /etc/fstab

192.168.1.51:/var/www/mysite/media /mnt/cloudstation/ nfs rw,async 0 0

192.168.1.51:/var/www/mysite/media /home/martin/Documents/dev_enaex/mnt/alarmas/ nfs rw,async 0 0

luego desmontar y montar con con:
sudo umount /mnt/cloudstation/
sudo mount -a

En el server 
sudo mount -o bind /var/www/mysite/media/alarmas /home/sepulm01/dev_enaex/mnt/alarmas

en el fstab:
sudo nano /etc/fstab
/var/www/mysite/media/alarmas /home/sepulm01/dev_enaex/mnt/alarmas auto bind 0 0
sudo mount -a

# Configuración detector mul rq:

Habilitar archivo conf.file en el mismo direcorio de dector_mul_rq.py
contenido tipo json:

{
 "website": "http://127.0.0.1:8000",
 "output_dir": "mysite/media/alarmas/"
}

# Visualizar videos

en el servidor web instalar ffmpeg

sudo apt-get ffmpeg


# Nvidia

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices

== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00002187sv00001043sd00008769bc03sc00i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-440-server - distro non-free
driver   : nvidia-driver-450 - third-party free recommended
driver   : xserver-xorg-video-nouveau - distro free builtin

sudo apt install nvidia-driver-440-server
