# package-routine-web

Web application to select MAGMA routines, and then download a tarfile containing all the source code required for it.

Install the requirements with `pip3 install -r requirements.txt`

Run the application with `flask run`, or you can use `python3 app.py` (but the first is preferred, since you can set hostname/port)

## Generating Tarfiles

Use the `tools/package-routine.py` to generate a single tarfile, and store it in this directory `/tars` and then in the appropriate folder for each backend (even though the backend largely doesn't matter)

