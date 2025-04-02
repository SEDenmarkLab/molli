// This is where we define functions pertaining to 3dmol.js interaction with molli

async function get_molecules() {

    var keys = [];
    var opts = document.getElementById("molid").selectedOptions;

    for (let i = 0; i < opts.length; i++) {
        keys.push(opts[i].label);
    }

    let data_post = { get_items: { keys: keys, fmt: "sdf" } };

    response = await fetch(window.location.href, {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data_post)
    })

    let data = await response.json()

    // let viewerDiv = document.getElementById('v3dm');
    let viewer = $3Dmol.viewers["v3dm"]
    viewer.removeAllModels()
    viewer.setStyle({ "line": { opacity: 0.3 } })

    for (const key in data) {
        let ms = viewer.addModels(data[key], "sdf");
        ms[0].setStyle({}, { "stick": { "radius": 0.1, opacity: 1 }, "sphere": { "scale": 0.11, opacity: 1 }, })
    }


    viewer.setBackgroundColor(0x000, 0)




    viewer.zoomTo();
    viewer.render();
}

async function get_keys() {
    const molid = document.getElementById("molid")

    response = await fetch(window.location.href, {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ get_keys: null })
    })

    let data = await response.json();
    keys = data["keys"]

    for (const key of keys) {
        let opt = document.createElement("option");
        opt.label = key;
        molid.add(opt);
    }

    select_first();
}

function filter_keys() {
    let wildcard = document.getElementById("wildcard").value

    if (wildcard == "") {
        wildcard = "*";
        document.getElementById("wildcard").value = "*"
    }
    // Any number of any chars
    wildcard = wildcard.replaceAll(/\*/g, ".*")

    const regex = new RegExp("^" + wildcard + "$");

    var opts = document.getElementById("molid").options;

    for (let i = 0; i < opts.length; i++) {
        if (regex.test(opts[i].label)) {
            opts[i].hidden = false
        }
        else {
            opts[i].hidden = true
        }
    }

    select_first()
}

function select_first() {
    let opts = document.getElementById("molid").options
    for (let i = 0; i < opts.length; i++) {
        if (!opts[i].hidden) {
            opts[i].selected = true;
            get_molecules().then();
            break;
        }
    }
}