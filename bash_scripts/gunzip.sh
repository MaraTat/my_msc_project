#!/bin/bash
for file in $1/*.ent.gz
do
        gunzip -q $file
done
echo All done
