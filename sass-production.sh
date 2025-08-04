#!/bin/bash

sass --load-path=node_modules ./scss/app.scss:./site/css/app.min.css --style compressed --no-source-map