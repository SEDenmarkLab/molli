import molli as ml
from pprint import pprint
from argparse import ArgumentParser
from pathlib import Path
from flask import Flask, request, abort, Response
from itertools import chain
from molli.chem.library import library_type

MOLLI_VERSION = ml.__version__

arg_parser = ArgumentParser(
    "molli show",
    description=__doc__,
)

arg_parser.add_argument(
    "path",
    action="store",
    nargs="?",
    help="Path from which the server content will be determined. If it's a directory",
)

arg_parser.add_argument(
    "--host",
    action="store",
    default="0.0.0.0",
    help="Host",
)

arg_parser.add_argument(
    "-p",
    "--port",
    action="store",
    default="15000",
    help="If the visualization protocol requires to fire up a server, this will be the port of choice.",
)

arg_parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode. USE WITH CAUTION (READ FLASK DOCUMENTATION). DO NOT USE IF YOU DO NOT KNOW **EXACTLY** WHAT THIS DOES.",
)

app = Flask("molli")


def get_aux_file(fn: str | Path):
    return Path(ml.aux.__file__).parent / "files" / fn


def get_library_info(fp: Path):
    import os

    fsize = os.path.getsize(fp) / 1e6  # Get file size in MB

    return library_type(fp), fsize


@app.route("/", methods=["POST", "GET"])
def molli_landing():
    cwd: Path = app.config["cwd"]

    if request.method == "POST":
        abort(404)
    else:
        html = get_aux_file("library_index.html").read_text()
        liblist = {}
        for lib in chain(cwd.glob("**/*.mlib"), cwd.glob("**/*.clib")):
            ltyp, lsiz = get_library_info(lib)

            shortpath = lib.relative_to(cwd).as_posix()
            key = lib.relative_to(cwd).parent.as_posix() + "/"
            if key not in liblist:
                liblist[key] = ""
            if ltyp is None:
                continue
            else:
                ltyp = ltyp.__name__
            liblist[
                key
            ] += f"""<div class="libitem" onclick="location.href='/lib/{shortpath}';"><a>{lib.name}</a><p>{ltyp} | {lsiz:0.1f} MB</p></div>"""

        listing = ""
        for k in liblist:
            listing += f"""<h2>{k}</h2><div class="libitemlist">{liblist[k]}</div>\n"""

        html = html.replace(r"{{ lib_list }}", listing)
        return html


@app.route("/lib/<path:fpath>", methods=["POST", "GET"])
def molli_show_library(fpath=None):
    cwd: Path = app.config["cwd"]

    full_path: Path = cwd / fpath
    if not (
        full_path.suffix in {".mlib", ".clib"}
        and full_path.exists()
        and full_path.is_file()
    ):
        print(f"Blocked {fpath}")
        abort(404)

    if (ltype := library_type(fpath)) is not None:
        library = ltype(fpath)

    if request.method == "POST":
        # That's for javascript to retrieve molecules
        data = request.get_json()

        assert len(data) == 1
        key = list(data)[0]

        match key:
            case "get_items":
                data = data["get_items"]
                keys = data["keys"]
                fmt = data["fmt"]

                with library.reading():
                    response = {
                        k: ml.dumps(library[k], fmt, writer="obabel") for k in keys
                    }

            case "get_keys":
                with library.reading():
                    response = {"keys": sorted(library.keys())}

        return response

    else:
        tp = get_aux_file("library_view.html")

        with open(tp) as f:
            template = f.read()

        return template


ALLOWED_FILES = {
    "molli.svg": "image/svg+xml",
    "molli_3dmol.js": "text/javascript",
    "style1.css": "text/css",
}


@app.route("/<file>")
def get_file(file=None):
    match file:
        case "favicon.ico":
            abort(404)

        case _ if file in ALLOWED_FILES:
            with get_aux_file(file).open("rb") as f:
                return Response(f.read(), mimetype=ALLOWED_FILES[file])

        case _:
            return None


def molli_main(args, **kwargs):
    parsed = arg_parser.parse_args(args)

    if parsed.path is None:
        app.config["cwd"] = Path.cwd().absolute()
    else:
        app.config["cwd"] = Path(parsed.path).absolute()

    app.run(host=parsed.host, port=parsed.port, debug=parsed.debug)
