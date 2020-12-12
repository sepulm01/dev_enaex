#leellaves
import json
import Crypto
from Crypto.PublicKey import RSA
import binascii 
from Crypto.Cipher import PKCS1_OAEP
from datetime import datetime, timedelta


with open('private_key.txt') as json_file:
    data = json.load(json_file)
    private_key =  data['private_key']

with open('public_key.txt') as json_file:
    data = json.load(json_file)
    public_key =  data['public_key']

#print(public_key,'\n', private_key)

public_key = RSA.importKey(binascii.unhexlify(public_key))

private_key = RSA.importKey(binascii.unhexlify(private_key))
#public_key = private_key.publickey()

message = '2020-12-16T07:00:00Z'
message = message.encode()

cipher = PKCS1_OAEP.new(public_key)
encrypted_message  = cipher.encrypt(message)

#Bin to txt
e_m = binascii.hexlify(encrypted_message).decode('utf8')
print(e_m)
#txt to bin
zo = binascii.unhexlify(e_m)

cipher = PKCS1_OAEP.new(private_key)
message = cipher.decrypt(zo)
print(type(message), message)
#message=decode_binary_string(message)
message = message.decode('utf-8')
print(type(message), message )

d1 = datetime.strptime(message,"%Y-%m-%dT%H:%M:%SZ")

print(type(d1), d1 )
