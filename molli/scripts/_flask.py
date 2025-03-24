from flask import Flask, request
import molli as ml
from pathlib import Path


class _AppConfig:

    @property
    def libraries(self):
        return self._active_libraries

    @libraries.setter
    def libraries(self, other):
        self._active_libraries = other


app = Flask("molli")
config = _AppConfig()


@app.route("/<libname>", methods=["POST", "GET"])
def molli_library(libname=None):
    if libname not in config.libraries:
        return f"<p>Unknown library {libname}. Should be in: {sorted(config.libraries)}</p>"
    else:
        library = config.libraries[libname]

    # This actually processes the request

    if request.method == "POST":
        # That's for javascript to retrieve molecules
        data = request.get_json()

        keys = data["keys"]
        fmt = data["fmt"]

        with library.reading():
            response = {k: ml.dumps(library[k], fmt, writer="obabel") for k in keys}

        return response

    else:
        tp = Path(ml.aux.__file__).parent / "files" / "library_view.html"
        with open(tp) as f:
            template = f.read()
        # mol: ml.Molecule = ml.load(f"data/{fname}.xyz", parser="openbabel")
        # mol = ml.load(ml.files.dendrobine_mol2)
        # obabel_optimize(mol, coord_displace=0.1, inplace=True)
        # mol.add_implicit_hydrogens()
        # obabel_optimize(mol, coord_displace=0.001, inplace=True)

        with library.reading():
            keys = list(library.keys())
            ens = library[keys[0]]

        return template


@app.route("/<libname>/info", methods=["POST"])
def molli_library_info(libname=None):
    if libname not in config.libraries:
        return f"<p>Unknown library {libname}. Should be in: {sorted(config.libraries)}</p>"
    else:
        library = config.libraries[libname]

    if request.method == "POST":
        # That's for javascript to retrieve molecules
        data = request.get_json()

        response = {}
        with library.reading():
            for prop in data:
                match prop:
                    case "keys":
                        response[prop] = sorted(library.keys())

                    case _:
                        response[prop] = None

        return response
