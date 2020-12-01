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


# SYSTEMD (si systemd)

cd /etc/systemd/system
sudo nano worker_st7.service

[Unit]
Description=worker 7 - sistema de seguridad
After=docker.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
RuntimeMaxSec=21600
User=sepulm01
Environment=CAM=7
Environment=TZ=America/Santiago
WorkingDirectory=/home/sepulm01/dev_enaex/
ExecStart=/usr/bin/python3 /home/sepulm01/dev_enaex/detector_mul_rq.py

[Install]
WantedBy=multi-user.target


sudo systemctl start worker_st1

# Configuración RTSP cámara

Sintaxis string de enlace:
    rtsp://<usuario>:<pass>@<URL o IP>:<puerto>/live.sdp
    rtsp://visor:Experimento.29@192.168.0.104:554/live.sdp


# FFMPEG server

feed: 
ffmpeg -i 'rtsp://visor:Experimento.29@192.168.0.104:554/live.sdp' -r 25 http://localhost:8080/feed.ffm


server:
ffserver

/etc/ffserver.conf

HTTPPort 8080                      # Port to bind the server to
HTTPBindAddress 0.0.0.0
MaxHTTPConnections 2000
MaxClients 1000
MaxBandwidth 10000             # Maximum bandwidth per client
                               # set this high enough to exceed stream bitrate
CustomLog -

<Feed feed.ffm>
     File ./feed.ffm
     FileMaxSize 1g
     ACL allow 127.0.0.1
</Feed>

<Stream feed.webm>
     Format webm
     Feed feed.ffm
     VideoCodec libvpx
     VideoSize 320x240
     VideoFrameRate 30
     VideoBitRate 512
     VideoBufferSize 512
     NoAudio
     AVOptionVideo flags +global_header
     StartSendOnKey
</Stream>

<Stream status.html>            # Server status URL
   Format status
   # Only allow local people to get the status
   #ACL allow 192.168.0.1 192.168.0.255
   ACL allow localhost
   ACL allow 192.168.0.0 192.168.255.255
</Stream>

#<Redirect index.html>    # Just an URL redirect for index
#   # Redirect index.html to the appropriate site
#  URL /
#</Redirect>


pagina web 

<!DOCTYPE html>
<html><head><title>Live Cam</title></head>

<body>

<video width="320" height="240" autoplay muted>
  <source src="http://localhost:8080/feed.webm" type="video/webm">
Your browser does not support the video tag.
</video>



</body>
</html>


# SYSTEMD para FFServer

cd /etc/systemd/system
sudo nano FFServer_st.service

[Unit]
Description=FFserver St - sistema de seguridad
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=sepulm01
Environment=TZ=America/Santiago
ExecStart=/usr/bin/ffserver

[Install]
WantedBy=multi-user.target


sudo systemctl start FFServer_st
sudo systemctl enable FFServer_st


[Unit]
Description=FFMPEG transcoder service
After=FFServer_st.service

[Service]
User=sepulm01
# -video_size and -framerate should match the settings in ffserver.conf
ExecStart=/usr/bin/ffmpeg -i 'rtsp://visor:Experimento.29@192.168.0.104:554/live.sdp' -re http://localhost:8080/feed.ffm

[Install]
WantedBy=multi-user.target


#Celery
pip3 install -U Celery
pip3 install -U "celery[redis]"


## IP FIJA


```
cd /etc/netplan
sudo nano xxxx.yaml

# This is the network config written by 'subiquity'
#network:
#  ethernets:
#    enp3s0:
#      dhcp4: true
#  version: 2

# This file describes the network interfaces available on your system
# For more information, see netplan(5).
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
     dhcp4: false
     addresses: [10.187.70.11/24]
     gateway4: 10.187.70.1
     nameservers:
       addresses: [8.8.8.8]
  version: 2


sudo netplan generate
sudo netplan apply
```