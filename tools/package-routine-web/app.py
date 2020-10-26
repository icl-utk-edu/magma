#!/usr/bin/env python
''' app.py - main application for the package-routine-web project

Runs a Flask application that runs the tool to select routines

Example:

$ flask run
$ flask run --help

@author: Cade Brown <cade@utk.edu>
'''

# flask
import flask
from flask import Flask, request, send_file, render_template

# python stdlib
import time
import io
import os
import tarfile

app = Flask(__name__)

@app.route('/')
def index_():
    return render_template('index.html')

@app.route('/list')
def list_():
    # returns list of valid routines
    names = []
    for full in map(lambda x: x[:x.index('.')], filter(lambda x: '.tar' in x, os.listdir('tars/cuda'))):
        names.append(full.replace('magma_', ''))
    return ','.join(names)

@app.route('/get')
def get_():
    curtime = time.time()
    interface = request.args.get('interface', 'cuda')

    routines = {*filter(lambda x: x, request.args.get('routines').split(','))}
    out_name = 'magma_' + interface + '_' + '_'.join(routines) + '.tar.gz'

    # create in-memory buffer which is a tarfile
    out_buf = io.BytesIO()
    out_tar = tarfile.open(fileobj=out_buf, mode='w:gz')

    # manifest files which need to be merged instead of xor'd
    merged = {}

    merge_files = {key: set() for key in [
        'FUNCS.mf',
        'WARNINGS.mf',
        'BLAS.mf',
    ]}
    
    # wraps as 'addfile' acceptable parameters to `tarfile.addfile`
    def wrap(fname, src):
        oi = tarfile.TarInfo(fname)
        # fill in meta data (size is required; else everything is empty!)
        oi.size = len(src)
        oi.mtime = curtime

        # we need to return a TarInfo and a readable IO-like object (as if a file was opened)
        return oi, io.BytesIO(src)

    # go through tar files for each routines
    for r in routines:
        tf = tarfile.open('tars/' + interface + '/magma_' + r + '.tar.gz')
        for fl in tf:
            # extract and read as bytes
            ef = tf.extractfile(fl)
            src = ef.read()
            if fl.name.endswith('.mf'):
                # just combine unique lines
                merged[fl.name] = merged.get(fl.name, set()) | {*src.split(b'\n')}
            elif fl.name not in out_tar:
                # add source
                out_tar.addfile(*wrap(fl.name, src))

    # add merged files in as well
    for fl in merged:
        if fl not in out_tar:
            out_tar.addfile(*wrap(fl, b'\n'.join(merged[fl])))

    # finish the tar file
    out_tar.close()

    # now, reset position so that the 'send_file' function reads it like an open file
    out_buf.seek(0)
    return send_file(out_buf, mimetype='tar', as_attachment=True, attachment_filename=out_name)

if __name__ == '__main__':
    app.run()
