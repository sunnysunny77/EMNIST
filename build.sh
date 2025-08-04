#!/bin/bash

set -e

set -x

npm run build-css &

npm run build-js &

wait