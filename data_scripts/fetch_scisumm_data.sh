#!/bin/bash

SCISUMM_DATA_PATH=https://cs.stanford.edu/~myasu/projects/scisumm_net/scisummnet_release1.1__20190413.zip

cd data/
gdown $SCISUMM_DATA_PATH
zip -d scisummnet_release1.1__20190413.zip __MACOSX/\*
unzip scisummnet_release1.1__20190413.zip
rm scisummnet_release1.1__20190413.zip
mv scisummnet_release1.1__20190413 scisummnet
cd ..
