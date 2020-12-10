import Crypto
from Crypto.PublicKey import RSA
import binascii
import json

random_gen = Crypto.Random.new().read

private_key = RSA.generate(1024, random_gen)
public_key = private_key.publickey()


private_key = private_key.exportKey(format='DER')
public_key = public_key.exportKey(format='DER')

private_key = binascii.hexlify(private_key).decode('utf8')
public_key = binascii.hexlify(public_key).decode('utf8')

print(private_key,"\n", public_key)

data ={}
data['private_key']=private_key
with open('private_key.txt', 'w') as outfile:
    json.dump(data, outfile)

data ={}
data['public_key']=public_key
with open('public_key.txt', 'w') as outfile:
    json.dump(data, outfile)