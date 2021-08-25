#!/bin/bash

for i in {0..2}
do
    S3_DATA_PATH=s3://ai2-s2-scia11y/s2orc_cc_filtered/20200705v1/preprocessed/$i/
    LOCAL_DATA_PATH=data/s2orc/$i/
    aws s3 sync $S3_DATA_PATH $LOCAL_DATA_PATH
done
