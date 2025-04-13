import os
import shutil
import re
import argparse
import csv
from datetime import datetime

special_domains = {
    "absa": "absa.co.za",
    "adidas": "adidas.com",
    "adobe": "adobe.com",
    "airbnb": "airbnb.com",
    "alibaba": "alibaba.com",
    "aliexpress": "aliexpress.com",
    "allegro": "allegro.pl",
    "amazon": "amazon.com",
    "ameli_fr": "ameli.fr",
    "american_express": "americanexpress.com",
    "anadolubank": "anadolubank.com",
    "aol": "aol.com",
    "apple": "apple.com",
    "arnet_tech": "arnet.com",
    "aruba": "aruba.it",
    "att": "att.com",
    "azul": "voeazul.com.br",
    "bahia": "bahia.com.br",
    "banco_de_occidente": "bancodeoccidente.com.co",
    "banco_inter": "bancointer.com.br",
    "bankia": "bankia.es",
    "barclaycard": "barclaycard.co.uk",
    "barclays": "barclays.co.uk",
    "bbt": "bbt.com",
    "bcp": "viabcp.com",
    "bestchange": "bestchange.com",
    "blizzard": "blizzard.com",
    "bmo": "bmo.com",
    "bnp_paribas": "bnpparibas.com",
    "bnz": "bnz.co.nz",
    "boa": "bankofamerica.com",
    "bradesco": "bradesco.com.br",
    "bt": "bt.com",
    "caixa_bank": "caixabank.es",
    "canada": "canada.ca",
    "capital_one": "capitalone.com",
    "capitec": "capitecbank.co.za",
    "cathay_bank": "cathaybank.com",
    "cetelem": "cetelem.fr",
    "chase": "chase.com",
    "cibc": "cibc.com",
    "cloudconvert": "cloudconvert.com",
    "cloudns": "cloudns.net",
    "cogeco": "cogeco.ca",
    "commonwealth_bank": "commbank.com.au",
    "cox": "cox.com",
    "crate_and_barrel": "crateandbarrel.com",
    "cryptobridge": "crypto-bridge.org",
    "daum": "daum.net",
    "db": "db.com",
    "dhl": "dhl.com",
    "dkb": "dkb.de",
    "docmagic": "docmagic.com",
    "dropbox": "dropbox.com",
    "ebay": "ebay.com",
    "eharmony": "eharmony.com",
    "erste": "erstegroup.com",
    "etisalat": "etisalat.ae",
    "etrade": "etrade.com",
    "facebook": "facebook.com",
    "fibank": "fibank.bg",
    "file_transfer": "filetransfer.io",
    "fnac": "fnac.com",
    "fsnb": "fsnb.com",
    "godaddy": "godaddy.com",
    "google": "google.com",
    "gov_uk": "gov.uk",
    "grupo_bancolombia": "grupobancolombia.com",
    "hfe": "hfe.co.uk",
    "hsbc": "hsbc.com",
    "htb": "hackthebox.eu",
    "ics": "icscards.nl",
    "ieee": "ieee.org",
    "impots_gov": "impots.gouv.fr",
    "infinisource": "infinisource.com",
    "instagram": "instagram.com",
    "irs": "irs.gov",
    "itau": "itau.com.br",
    "knab": "knab.nl",
    "la_banque_postale": "labanquepostale.fr",
    "la_poste": "laposte.fr",
    "latam": "latam.com",
    "lbb": "lbb.de",
    "lcl": "lcl.fr",
    "linkedin": "linkedin.com",
    "lloyds_bank": "lloydsbank.com",
    "made_in_china": "made-in-china.com",
    "mbank": "mbank.pl",
    "mdpd": "miamidade.gov",
    "mew": "myetherwallet.com",
    "microsoft": "microsoft.com",
    "momentum_office_design": "momentumofficeonline.com",
    "mweb": "mweb.co.za",
    "my_cloud": "mycloud.com",
    "nab": "nab.com.au",
    "natwest": "natwest.com",
    "navy_federal": "navyfederal.org",
    "nedbank": "nedbank.co.za",
    "netflix": "netflix.com",
    "netsons": "netsons.com",
    "nordea": "nordea.com",
    "ocn": "ocn.ne.jp",
    "one_and_one": "1and1.com",
    "orange": "orange.com",
    "orange_rockland": "oru.com",
    "otrs": "otrs.com",
    "ourtime": "ourtime.com",
    "paschoalotto": "paschoalotto.com.br",
    "paypal": "paypal.com",
    "postbank": "postbank.de",
    "qnb": "qnb.com",
    "rbc": "rbcroyalbank.com",
    "runescape": "runescape.com",
    "sharp": "sharp.com",
    "shoptet": "shoptet.cz",
    "sicil_shop": "sicil.com",
    "smartsheet": "smartsheet.com",
    "smiles": "smiles.com.br",
    "snapchat": "snapchat.com",
    "sparkasse": "sparkasse.de",
    "standard_bank": "standardbank.co.za",
    "steam": "steampowered.com",
    "strato": "strato.de",
    "stripe": "stripe.com",
    "summit_bank": "summitbank.com",
    "sunrise": "sunrise.ch",
    "suntrust": "suntrust.com",
    "swisscom": "swisscom.ch",
    "taxact": "taxact.com",
    "tech_target": "techtarget.com",
    "telecom": "telecom.com",
    "test_rite": "testrite.com",
    "timeweb": "timeweb.com",
    "tradekey": "tradekey.com",
    "twins_bnk": "twinsbank.com",
    "twitter": "twitter.com",
    "typeform": "typeform.com",
    "usaa": "usaa.com",
    "walmart": "walmart.com",
    "wells_fargo": "wellsfargo.com",
    "whatsapp": "whatsapp.com",
    "wp60": "wordpress.com",
    "xtrix_tv": "xtrix.tv",
    "yahoo": "yahoo.com",
    "youtube": "youtube.com",
    "ziggo": "ziggo.nl",
    "google_drive": "drive.google.com",
    "ms_skype": "skype.com",
    "ms_onedrive": "onedrive.live.com",
    "ms_outlook": "outlook.com",
    "ms_bing": "bing.com",
    "ms_office": "office.com",
    "itunes": "itunes.apple.com",
    "icloud": "icloud.com",
    "zoominfo": "zoominfo.com"
}


def sanitize_filename(filename):
    """Convert a string to a valid filename by removing invalid characters."""
    # Replace characters that are not allowed in filenames
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


def parse_folder(folder_path, quiet_mode=False, csv_path=None):
    """
    Parse a folder and restructure its subfolders according to the requirements.

    Args:
        folder_path: Path to the main folder (folder A)
        quiet_mode: If True, only print filepaths with no additional output
        csv_path: If provided, path where to save the CSV with old and new image paths
    """
    # Reverse the dictionary to look up by key
    reverse_lookup = {v: k for k, v in special_domains.items()}

    # Prepare CSV data if csv_path is provided
    csv_data = []

    # Get all immediate subfolders in folder A
    sub_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # Keep track of folders to remove later
    folders_to_remove = []

    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder_path, sub_folder)

        # Step 1: Rename the subfolder if it exists in the special_domains
        if sub_folder in reverse_lookup:
            new_folder_name = special_domains[reverse_lookup[sub_folder]]
        elif sub_folder in special_domains:
            new_folder_name = special_domains[sub_folder]
        else:
            # If the folder name is not in the dictionary, keep the original name
            new_folder_name = sub_folder

        # Create a sanitized version of the new folder name
        new_folder_name = sanitize_filename(new_folder_name)
        new_folder_path = os.path.join(folder_path, new_folder_name)

        # Rename the folder if the name has changed
        if new_folder_name != sub_folder:
            try:
                os.rename(sub_folder_path, new_folder_path)
                if not quiet_mode:
                    print(f"Renamed folder: {sub_folder} -> {new_folder_name}")
                else:
                    print(new_folder_path)
                sub_folder_path = new_folder_path  # Update the path for subsequent operations
            except Exception as e:
                if not quiet_mode:
                    print(f"Error renaming folder {sub_folder}: {e}")
                continue

        # Step 2: Process image files in the subfolder
        image_files = []
        for file in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(file)

        # Create new folders for each image and move the image
        for image_file in image_files:
            # Get the image name without extension
            image_name = os.path.splitext(image_file)[0]

            # Create a new folder name with the pattern "renamed_folder--ImageName"
            new_image_folder_name = f"{new_folder_name}--{image_name}"
            new_image_folder_name = sanitize_filename(new_image_folder_name)
            new_image_folder_path = os.path.join(folder_path, new_image_folder_name)

            # Create the new folder
            try:
                os.makedirs(new_image_folder_path, exist_ok=True)
                if not quiet_mode:
                    print(f"Created folder: {new_image_folder_name}")
                else:
                    print(new_image_folder_path)

                # Move the image to the new folder
                old_image_path = os.path.join(sub_folder_path, image_file)
                new_image_path = os.path.join(new_image_folder_path, image_file)

                # Store paths for CSV before moving
                if csv_path:
                    csv_data.append({
                        "old_path": old_image_path,
                        "new_path": new_image_path
                    })

                shutil.move(old_image_path, new_image_path)
                if not quiet_mode:
                    print(f"Moved image: {image_file} to {new_image_folder_name}")
            except Exception as e:
                if not quiet_mode:
                    print(f"Error processing image {image_file}: {e}")

        # Add this folder to the list of folders to be removed
        folders_to_remove.append(sub_folder_path)

    # Step 3: Remove empty folders
    for folder in folders_to_remove:
        try:
            # Check if folder is empty
            if os.path.exists(folder) and not os.listdir(folder):
                os.rmdir(folder)
                if not quiet_mode:
                    print(f"Removed empty folder: {folder}")
        except Exception as e:
            if not quiet_mode:
                print(f"Error removing folder {folder}: {e}")

    # Write CSV if path is provided
    if csv_path and csv_data:
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['old_path', 'new_path']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in csv_data:
                    writer.writerow(row)

            if not quiet_mode:
                print(f"CSV file with paths written to: {csv_path}")
        except Exception as e:
            if not quiet_mode:
                print(f"Error writing CSV file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a folder and reorganize its subfolders and images.')
    parser.add_argument('folder_path', help='Path to the main folder to process')
    parser.add_argument('--quiet', '-q', action='store_true', help='Print only filepaths without additional output')
    parser.add_argument('--csv', '-c', help='Path to save a CSV file with old and new image paths')

    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Error: {args.folder_path} is not a valid directory")
        exit(1)

    if not args.quiet:
        print(f"Processing folder: {args.folder_path}")

    # If csv path is not specified but the flag is used, create a default name
    csv_path = args.csv
    if csv_path is True:  # This happens if --csv is used without a value
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"image_paths_{timestamp}.csv"

    parse_folder(args.folder_path, args.quiet, csv_path)

    if not args.quiet:
        print("Processing complete!")
