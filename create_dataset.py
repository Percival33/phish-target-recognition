#!/usr/bin/env python3
"""
Simple dataset creation script for phishing target recognition.
Creates symlinked datasets from phishpedia data in VisualPhish format.

Usage:
    uv run create_dataset.py \
        --benign-dir /path/to/benign/folders \
        --phishing-dir /path/to/phishing/folders \
        --output-dir /path/to/output \
        --format visualphish
"""

import json
import os
import argparse
import pickle
import sys
from pathlib import Path


def get_special_domain_mapping():
    """Fallback domain mapping from organize.py"""
    return {
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


def load_domain_mapping():
    """Load target name mappings from pickle file or fallback to hardcoded mapping."""
    try:
        with open('src/models/phishpedia/models/domain_map.pkl', 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, PermissionError):
        print("Warning: Could not load domain_map.pkl, using fallback mapping")
        return get_special_domain_mapping()


def save_visualphish_format(target_num, image_paths, output_dir, is_phishing=False):
    """Save images in VisualPhish format using symlinks."""
    subdir = "phishing" if is_phishing else "benign"
    output_path = Path(output_dir) / subdir
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for i, img_path in enumerate(image_paths):
        if Path(img_path).exists():
            link_name = f"T{target_num}_{i}.png"
            link_path = output_path / link_name
            
            # Remove existing symlink if it exists
            if link_path.exists():
                link_path.unlink()
            
            os.symlink(str(Path(img_path).absolute()), str(link_path))
            count += 1
    
    return count


def find_benign_images(domains, benign_dir):
    """Find folders matching domains and return shot.png paths."""
    image_paths = []
    benign_path = Path(benign_dir)
    
    if not benign_path.exists():
        print(f"Warning: Benign directory {benign_dir} does not exist")
        return image_paths
    
    for domain in domains:
        domain = domain.strip()
        if not domain:
            continue
            
        domain_folder = benign_path / domain
        if domain_folder.exists() and domain_folder.is_dir():
            shot_png = domain_folder / "shot.png"
            if shot_png.exists():
                image_paths.append(str(shot_png))
            else:
                print(f"Warning: No shot.png found in {domain_folder}")
        else:
            print(f"Warning: Domain folder {domain_folder} not found")
    
    return image_paths


def find_phishing_images(target_name, phishing_dir, domain_mapping):
    """Map target name to company name and find matching folders."""
    image_paths = []
    phishing_path = Path(phishing_dir)
    
    if not phishing_path.exists():
        print(f"Warning: Phishing directory {phishing_dir} does not exist")
        return image_paths
    
    # Try to map target name to known company names
    target_lower = target_name.lower().replace(" ", "_")
    
    # Create reverse mapping from company names to target keys
    reverse_mapping = {}
    for key, domain in domain_mapping.items():
        reverse_mapping[key] = key
    
    # Special cases for target name mapping
    target_mappings = {
        "gov_uk": "Government of the United Kingdom",
        "capital_one": "Capital One",
        "banco_inter": "Banco Inter",
        "american_express": "American Express",
        "linkedin_corporation": "LinkedIn",
        "made-in-china": "Made-In-China",
        "natwest_personal_banking": "NatWest",
        "navy_federal_credit_union": "Navy Federal Credit Union",
        "office365": "Microsoft",
        "paypal": "PayPal",
        "runescape": "RuneScape",
        "sparkasse_bank": "Sparkasse",
        "steam": "Steam",
        "twitter,_inc": "Twitter",
        "visa_international_service_association": "Visa",
        "yahoo!": "Yahoo",
        "ebay": "eBay",
    }
    
    # Try different variations of the target name
    search_names = [
        target_name,
        target_mappings.get(target_lower, target_name),
        target_name.replace("_", " "),
        target_name.replace(" ", "_"),
    ]
    
    for search_name in search_names:
        # Find folders that start with the search name
        for folder in phishing_path.iterdir():
            if folder.is_dir() and folder.name.startswith(search_name + "+"):
                shot_png = folder / "shot.png"
                if shot_png.exists():
                    image_paths.append(str(shot_png))
                else:
                    print(f"Warning: No shot.png found in {folder}")
        
        if image_paths:  # Found some matches, stop searching
            break
    
    if not image_paths:
        print(f"Warning: No phishing folders found for target '{target_name}'")
    
    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset from phishpedia data using symlinks"
    )
    parser.add_argument("--benign-dir", required=True, help="Directory containing benign domain folders")
    parser.add_argument("--phishing-dir", required=True, help="Directory containing phishing target folders")
    parser.add_argument("--output-dir", required=True, help="Output directory for dataset")
    parser.add_argument("--format", default="visualphish", choices=["visualphish"], help="Output format")
    parser.add_argument("--json-file", default="mappings/pp-benign-trusted-logos-targets.json", help="JSON file with target definitions")
    
    args = parser.parse_args()
    
    # Load JSON file and filter phish=true entries
    try:
        with open(args.json_file) as f:
            all_targets = json.load(f)
        targets = [t for t in all_targets if t.get("phish")]
        print(f"Loaded {len(targets)} targets with phish=true from {args.json_file}")
    except FileNotFoundError:
        print(f"Error: JSON file {args.json_file} not found")
        return 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {args.json_file}")
        return 1
    
    # Load domain mapping
    domain_mapping = load_domain_mapping()
    print(f"Loaded domain mapping with {len(domain_mapping)} entries")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each target
    total_benign = 0
    total_phishing = 0
    
    for i, target in enumerate(targets):
        print(f"\nProcessing target {i}: {target['target']}")
        
        # Find benign images by domain
        domains = target["domains"].split("\n") if target.get("domains") else []
        benign_images = find_benign_images(domains, args.benign_dir)
        benign_count = save_visualphish_format(i, benign_images, args.output_dir, is_phishing=False)
        total_benign += benign_count
        
        # Find phishing images by target name
        phishing_images = find_phishing_images(target["target"], args.phishing_dir, domain_mapping)
        phishing_count = save_visualphish_format(i, phishing_images, args.output_dir, is_phishing=True)
        total_phishing += phishing_count
        
        print(f"  → {benign_count} benign, {phishing_count} phishing images")
    
    print(f"\n=== Summary ===")
    print(f"Processed {len(targets)} targets")
    print(f"Created {total_benign} benign symlinks")
    print(f"Created {total_phishing} phishing symlinks")
    print(f"Output directory: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

