#!/bin/bash

# Get the directory where this script is located, and cd to it
THISDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export environment variable
echo "Exporting GWSAMPLEGEN_DIR=${THISDIR}"
export GWSAMPLEGEN_DIR="${THISDIR}"

# Append to ~/.bashrc if not already present
if ! grep -q "export GWSAMPLEGEN_DIR=" ~/.bashrc; then
	echo "Appending GWSAMPLEGEN_DIR to ~/.bashrc"
	echo "export GWSAMPLEGEN_DIR=\"${THISDIR}\"" >> ~/.bashrc
else
	echo "GWSAMPLEGEN_DIR already set in ~/.bashrc. Overwriting..."
	#overwrite the existing line to ensure it points to the correct directory
	sed -i "s|^export GWSAMPLEGEN_DIR=.*|export GWSAMPLEGEN_DIR=\"${THISDIR}\"|" ~/.bashrc
	echo "Updated GWSAMPLEGEN_DIR in ~/.bashrc"
fi

#add user input to confirm that they want to install into the current environment
read -p "Do you want to install GWSamplegen into the current environment? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
	echo "Installation halted. Please ensure you have a Python virtual environment available for installing ."
	exit 1
fi

pip install .

echo "Installation complete. Please restart your terminal or run 'source ~/.bashrc' to update your environment."
