#!/bin/bash
# Installs wkthmltopdf package for Linux-based systems with amd64 architecture.
# For other systems and architectures, refer to https://wkhtmltopdf.org/downloads.html
#
# Usage: ./install_wkhtmltopdf.sh
# -----------------------------------------------------------------------------

wget --no-check-certificate https://downloads.wkhtmltopdf.org/0.12/0.12.5/wkhtmltox_0.12.5-1.bionic_amd64.deb
sudo dpkg -i wkhtmltox_0.12.5-1.bionic_amd64.deb
sudo apt-get install -f
rm wkhtmltox_0.12.5-1.bionic_amd64.deb