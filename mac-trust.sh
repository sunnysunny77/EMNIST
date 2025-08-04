#!/bin/bash

source $INIT_CWD/.env

sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain $INIT_CWD/certs/ca.crt 