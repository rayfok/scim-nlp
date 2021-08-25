#!/bin/bash

IUI_DATA_PATH=https://drive.google.com/uc?id=1x7yFlB0iOsg-8zv4ek5rUEWEqsZpNl9x

cd data/
gdown $IUI_DATA_PATH
unzip iui_19-21.zip
rm iui_19-21.zip
cd ..
