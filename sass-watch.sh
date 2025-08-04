#!/bin/bash

sass --load-path=node_modules --embed-sources ./scss/app.scss:./site/css/app.min.css --style compressed -w