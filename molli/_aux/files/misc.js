function change_color_scheme(scheme) {
    localStorage.setItem('molli_libview_color_scheme', scheme)
    document.documentElement.style.colorScheme = scheme
}

function recall_color_scheme() {
    let scheme = localStorage.getItem('molli_libview_color_scheme');
    document.documentElement.style.colorScheme = scheme
}