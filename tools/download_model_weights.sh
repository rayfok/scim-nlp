#!/bin/bash
if [ ! -d models/ ]
then
    wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/scienceparseplus/models/latest.zip -O latest.zip
    unzip latest.zip 
    rm latest.zip
    echo "Successfully downloaded model weights."
else
    echo "The model folder already exists."
fi

wget https://maven.ceon.pl/artifactory/kdd-releases/pl/edu/icm/cermine/cermine-impl/1.13/cermine-impl-1.13-jar-with-dependencies.jar -O cermine-impl-1.13-jar-with-dependencies.jar

echo "Successfully downloaded CERMINE engine."