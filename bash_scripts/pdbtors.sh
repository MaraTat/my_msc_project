#!/bin/bash

for file in $1/*.ent
do 
	mkdir -p $1/torsions
	pdbtorsions $file > $1/torsions/$( basename -s .ent $file ).txt
done

echo All done

