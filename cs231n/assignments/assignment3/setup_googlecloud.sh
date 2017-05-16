#!/usr/bin/env bash

# This is the set-up script for Google Cloud.
sudo apt-get update
sudo apt-get install --yes libncurses5-dev
sudo apt-get install --yes python-dev
sudo apt-get install --yes python-pip
sudo apt-get install --yes libjpeg8-dev
sudo ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
conda install --yes pillow
sudo apt-get build-dep python-imaging
sudo apt-get install --yes libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
conda install --yes --file requirements.txt  # Install dependencies
echo "**************************************************"
echo "*****  End of Google Cloud Set-up Script  ********"
echo "**************************************************"
echo ""
