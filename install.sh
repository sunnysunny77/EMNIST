#!/bin/bash

set -e

source $INIT_CWD/.env

if [ ! -f $INIT_CWD/conf/ca.conf ] && \
   [ ! -f $INIT_CWD/conf/csr.conf ] && \
   [ ! -f $INIT_CWD/conf/cert.conf ] && \
   [ ! -f $INIT_CWD/certs/ca.key ] && \
   [ ! -f $INIT_CWD/certs/ca.crt ] && \
   [ ! -f $INIT_CWD/certs/ca.srl ] && \
   [ ! -f $INIT_CWD/certs/server.csr ] && \
   [ ! -f $INIT_CWD/certs/server.crt ] && \
   [ ! -f $INIT_CWD/certs/server.key ]; then

echo "Create certificates"

echo -e "[ req ]
prompt = no\n\
distinguished_name = dn\n\
x509_extensions = v3_ca\n\
[ dn ]\n\
C = $C\n\
ST = $ST\n\
L = $L\n\
O = $O\n\
OU = $OU\n\
CN = $CN Root CA\n\
[ v3_ca ]\n\
subjectKeyIdentifier=hash\n\
authorityKeyIdentifier=keyid:always,issuer:always\n\
#basicConstraints = critical,CA:true\n\
basicConstraints = CA:true" > $INIT_CWD/conf/ca.conf 

echo -e "[ req ]\n\
prompt = no\n\
req_extensions = req_ext\n\
distinguished_name = dn\n\
[ dn ]\n\
C = $C\n\
ST = $ST\n\
L = $L\n\
O = $O\n\
OU = $OU\n\
CN = $CN\n\
[ req_ext ]\n\
subjectAltName = @alt_names\n\
[ alt_names ]\n\
DNS.1 = $CN\n\
DNS.2 = localhost\n\
IP.1 = 127.0.0.1"> $INIT_CWD/conf/csr.conf
 
echo -e "authorityKeyIdentifier=keyid,issuer\n\
basicConstraints=CA:FALSE\n\
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment\n\
subjectAltName = @alt_names\n\
[alt_names]\n\
DNS.1 = $CN\n\
DNS.2 = localhost\n\
IP.1 = 127.0.0.1"> $INIT_CWD/conf/cert.conf

openssl genrsa \
-des3 \
-passout pass:$PASSWORD \
-out $INIT_CWD/certs/ca.key 2048

openssl req \
-x509 \
-new \
-sha256 \
-days 1024 \
-out $INIT_CWD/certs/ca.crt \
-nodes  \
-key $INIT_CWD/certs/ca.key \
-passin pass:$PASSWORD  \
-config $INIT_CWD/conf/ca.conf \

openssl req \
-new \
-sha256 \
-nodes \
-out $INIT_CWD/certs/server.csr \
-newkey rsa:2048 \
-keyout $INIT_CWD/certs/server.key \
-config $INIT_CWD/conf/csr.conf 

openssl x509 \
-req \
-in $INIT_CWD/certs/server.csr \
-CA $INIT_CWD/certs/ca.crt \
-CAkey $INIT_CWD/certs/ca.key \
-passin pass:$PASSWORD \
-CAcreateserial \
-out $INIT_CWD/certs/server.crt \
-days 500 \
-sha256 \
-extfile $INIT_CWD/conf/cert.conf

echo "Trust certificates"

npm run trust-cert

echo "Trust hosts"

npm run trust-hosts

else

echo "Certificate detected, deleted all files in conf and certs folders";

fi