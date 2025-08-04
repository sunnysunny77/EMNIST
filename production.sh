#!/bin/bash

set -e

set -x

npm run production-css &

npm run production-js &

wait