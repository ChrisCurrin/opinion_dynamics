#!/bin/bash
sphinx-apidoc -f -o source/ ../
make html
