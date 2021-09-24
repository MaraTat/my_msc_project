#!/bin/bash

for file in $1/*.zip
do
        mkdir -p $1/unzipped/
        unzip -q $file
        mv *.ent.gz $1/unzipped
done
echo All done
