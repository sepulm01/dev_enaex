# dev_enaex


# Volumen NFS

  sudo apt install nfs-common

sudo docker run                                            \
  -v /home/martin/Documents/dev_enaex/mysite/static:/static  \
  -e NFS_EXPORT_0='/static                  *(rw,no_subtree_check)' \
  --privileged                                 \
  -p 2049:2049                                        \
  erichough/nfs-server


  mount 0.0.0.0:/static /tmp/static