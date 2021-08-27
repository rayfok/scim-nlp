#!/bin/bash

S2ORC_ACL_DATA_PATH=https://drive.google.com/uc?id=1bn_FhUDfIzvOSyn-SRSaG2tosG9kVhrw

cd data/
gdown $S2ORC_ACL_DATA_PATH
tar -xvf s2orc_acl_20200705v1.tar.gz
rm s2orc_acl_20200705v1.tar.gz
cd ..
