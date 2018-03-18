#!/usr/bin/env bash

# Build
./build.sh

# Push on PyPi
echo "Publish on PyPi..."
until twine upload dist/*
do
  echo "Try again to login on PyPI and release this library..."
  read -p "Press Enter to continue > "
done

echo "Now update the file ../requirements.txt"
