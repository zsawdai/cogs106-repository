#!/bin/bash

git status
git pull
now=$(date)
echo "$now" > version
git add version
git add update-version.sh
git commit -m "Updated"  
git push
git status
