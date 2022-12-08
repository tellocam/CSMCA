#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wrapper script to send a CUDA source file to the remote exercise environment and retrieve the results.

   Usage: ./csmca.py FILE.cu ARG0 ARG1 ARG2 ...

   where
     FILE.cu is the respective source file with your code
     ARG0 ARG1 ARG2 ... are user-supplied arguments and will be passed to the executable.

   If you have any suggestions for improvements of this script, please contact:

   Author: Karl Rupp <rupp@iue.tuwien.ac.at>
   Course: Computational Science on Many Core Architectures, 360252, TU Wien

   ---

   Copyright 2022, Karl Rupp

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""



import requests
import sys

#url = 'https://k40.360252.org/2022/ex7/run.php'
url = 'https://rtx3060.360252.org/2022/ex7/run.php'


# Check whether a source file has been passed:
if len(sys.argv) < 2:
  print("ERROR: No source file specified")
  sys.exit()

# Read source file contents:
try:
  src_file = open(sys.argv[1], "r")
except FileNotFoundError:
  print('ERROR: Source file does not exist!')
  sys.exit()
sources = src_file.read()


# Set up JSON object to hold the respective fields, then send to the server and print the returned output (strip HTML tags, don't repeat the source code)
myobj = {'src': sources,
         'userargs': ' '.join(sys.argv[2:]),
         'grind': 'none',   # Possible values: none, valgrind, valgrindfull, memcheck, racecheck, synccheck, initcheck
         'profiler': 'none'} # Possible values: none, nvprof

response = requests.post(url, data = myobj)

print("POSTing request to " + url + "\n")
print(response.text.split("</pre><h1>")[1].replace("<h1>","").replace("</h1>","").replace("<pre>","").replace("</pre>","").replace("<br />","\n").replace("</html>",""))