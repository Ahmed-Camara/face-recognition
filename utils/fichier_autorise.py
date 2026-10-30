ALLOWED_EXTENSIONS = {"jpeg","jpg","png"}

def fichier_autorise(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS