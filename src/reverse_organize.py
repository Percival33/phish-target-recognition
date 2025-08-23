#!/usr/bin/env python3
"""
Script to reverse domain-based folder organization back to target names.
Processes folders named as domains, maps them to target names, and renames files.

Usage: python reverse_organize.py /path/to/domain_folders /path/to/output [--dry-run]
"""

import os
import sys
import shutil
import re
import argparse
import pandas as pd


def get_special_domain_mapping():
    """Return a dictionary mapping companies to their special domain codes."""
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
        "mdpd": "mps.it",
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


def build_inverse_mapping(forward_map):
    """Invert mapping, ensuring no domain collisions."""
    inverse = {}
    for label, domain in forward_map.items():
        if domain in inverse and inverse[domain] != label:
            raise ValueError(f"Domain '{domain}' maps to multiple labels")
        inverse[domain] = label
    return inverse


def process_domain_folder(
    domain_name, input_dir, output_dir, inverse_mapping, csv_records, dry_run=False
):
    """Process a single domain folder and copy its files to the target folder."""
    # Get target name from domain
    if domain_name not in inverse_mapping:
        raise ValueError(f"Domain '{domain_name}' not found in inverse mapping")

    target_name = inverse_mapping[domain_name]
    input_path = os.path.join(input_dir, domain_name)
    output_path = os.path.join(output_dir, target_name)

    # Get image files from input domain folder
    image_exts = {".jpg", ".jpeg", ".png"}

    def is_image(fname):
        return (
            os.path.isfile(os.path.join(input_path, fname))
            and os.path.splitext(fname)[1].lower() in image_exts
        )

    input_imgs = sorted([f for f in os.listdir(input_path) if is_image(f)])
    if not input_imgs:
        print(f"No images found in {input_path}")
        return

    # Create output directory if it doesn't exist (only if not dry run)
    if not dry_run:
        os.makedirs(output_path, exist_ok=True)

    # Find existing T number and max N for that T number
    T_re = re.compile(r"^T(\d+)_\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    Tnum = 1
    N_start = 0

    if os.path.exists(output_path):
        T_numbers = [
            int(m.group(1))
            for fname in os.listdir(output_path)
            if (m := T_re.match(fname))
        ]

        if T_numbers:
            # All files in a target folder should have the same T number
            # Verify consistency and use the existing T number
            unique_T = set(T_numbers)
            if len(unique_T) > 1:
                raise ValueError(
                    f"Inconsistent T numbers found in {output_path}: {unique_T}"
                )
            Tnum = T_numbers[0]  # Use existing T number

            # Find max N for this T number
            N_re = re.compile(rf"^T{Tnum}_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
            N_used = [
                int(m.group(1))
                for fname in os.listdir(output_path)
                if (m := N_re.match(fname))
            ]
            N_start = max(N_used) + 1 if N_used else 0
        else:
            # No existing T-numbered files, start with T1
            Tnum = 1
            N_start = 0

    # Process each image
    for idx, imgname in enumerate(input_imgs):
        ext = os.path.splitext(imgname)[1]
        destfname = f"T{Tnum}_{N_start + idx}{ext.lower()}"
        source_path = os.path.join(input_path, imgname)
        dest_path = os.path.join(output_path, destfname)

        if dry_run:
            print(
                f"[DRY RUN] Would copy: {domain_name}/{imgname} -> {target_name}/{destfname}"
            )
        else:
            if os.path.exists(dest_path):
                raise FileExistsError(
                    f"Destination file already exists and cannot be overwritten: {dest_path}"
                )
            shutil.copy2(source_path, dest_path)
            print(f"Copied: {domain_name}/{imgname} -> {target_name}/{destfname}")

        # Add record to CSV data
        csv_records.append(
            {"target": target_name, "input_name": imgname, "output_name": destfname}
        )


def main():
    parser = argparse.ArgumentParser(
        description="Reverse domain-based folder organization to target names."
    )
    parser.add_argument("input_dir", help="Directory containing domain folders")
    parser.add_argument(
        "output_dir", help="Directory where target folders will be created"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files",
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
        raise FileNotFoundError(
            f"Input directory '{args.input_dir}' does not exist or is not a directory."
        )

    # Build inverse mapping
    forward_mapping = get_special_domain_mapping()
    inverse_mapping = build_inverse_mapping(forward_mapping)

    # Setup CSV file
    csv_file = os.path.join(args.output_dir, "file_operations.csv")
    csv_records = []

    # Get all domain folders in input directory
    domain_folders = [
        d
        for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ]

    if not domain_folders:
        print(f"No domain folders found in {args.input_dir}")
        return

    # Process each domain folder
    for domain_folder in domain_folders:
        print(f"Processing domain: {domain_folder}")
        try:
            process_domain_folder(
                domain_folder,
                args.input_dir,
                args.output_dir,
                inverse_mapping,
                csv_records,
                args.dry_run,
            )
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Save to CSV (even in dry run mode to preview what would be saved)
    if csv_records:
        if args.dry_run:
            print(
                f"\n[DRY RUN] Would record {len(csv_records)} file operations in {csv_file}"
            )
        else:
            df = pd.DataFrame(csv_records)
            if os.path.exists(csv_file):
                # Append to existing CSV
                df.to_csv(csv_file, mode="a", header=False, index=False)
            else:
                # Create new CSV with headers
                df.to_csv(csv_file, index=False)
            print(f"\nRecorded {len(csv_records)} file operations in {csv_file}")
    else:
        print("No files were processed.")


if __name__ == "__main__":
    main()
