#!/usr/bin/env python3
"""
Script to organize images into folders based on labels and special domain mapping.

Usage: python organize_images.py /path/to/images /path/to/labels.txt /path/to/output
"""

import os
import sys
import shutil
import argparse


def load_labels(labels_file):
    """Load company labels from the labels.txt file."""
    with open(labels_file, "r") as f:
        # Strip whitespace and commas from each line
        return [line.strip().rstrip(",") for line in f if line.strip()]


def get_special_domain_mapping():
    """Return a dictionary mapping companies to their special domain codes."""
    # This is the special mapping dictionary - add more mappings as needed
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
        "zoominfo": "zoominfo.com",
    }
    return special_domains


def organize_images(images_dir, labels_file, output_dir):
    """
    Organize images into folders based on labels and special domain mapping.

    Args:
        images_dir: Directory containing numbered image files (000.png, 001.png, etc.)
        labels_file: Path to labels.txt containing company names
        output_dir: Directory where the organized folder structure will be created
    """
    # Load labels
    labels = load_labels(labels_file)

    # Get special domain mapping
    special_domains = get_special_domain_mapping()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files sorted numerically
    image_files = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if (f.endswith(".png") or f.endswith(".PNG") or f.endswith(".jpg"))
            and f.split(".")[0].isdigit()
        ],
        key=lambda x: int(x.split(".")[0]),
    )

    # Check if we have enough labels for the images
    if len(image_files) > len(labels):
        print(
            f"Warning: Found {len(image_files)} images but only {len(labels)} labels."
        )
        print("Some images will not be processed.")

    # Process each image with corresponding label
    for i, image_file in enumerate(image_files):
        if i >= len(labels):
            break

        # Get the corresponding label
        label = labels[i].lower()

        # Determine the folder code
        folder_code = special_domains.get(label, label[:2])

        # Create subdirectory
        subdir_path = os.path.join(output_dir, folder_code)
        os.makedirs(subdir_path, exist_ok=True)

        # Copy the image to the new location
        source_path = os.path.join(images_dir, image_file)
        dest_path = os.path.join(subdir_path, image_file)

        shutil.copy(source_path, dest_path)
        print(f"Copied {image_file} to {os.path.join(folder_code, image_file)}")

    print(f"\nProcessed {min(len(image_files), len(labels))} images.")


def main():
    parser = argparse.ArgumentParser(
        description="Organize images into folders based on labels."
    )
    parser.add_argument("images_dir", help="Directory containing the images")
    parser.add_argument("labels_file", help="Path to labels.txt file")
    parser.add_argument(
        "output_dir", help="Directory where to create the folder structure"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    # Check if directories and files exist
    if not os.path.isdir(args.images_dir):
        print(
            f"Error: Images directory '{args.images_dir}' does not exist",
            file=sys.stderr,
        )
        return 1

    if not os.path.isfile(args.labels_file):
        print(
            f"Error: Labels file '{args.labels_file}' does not exist", file=sys.stderr
        )
        return 1

    # Run the organization function
    organize_images(args.images_dir, args.labels_file, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
