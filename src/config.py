from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]


class AppConfig(BaseSettings):
    DATA_DIR: Path = PROJ_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    PHISHIRIS_DL_DATASET_DIR: Path = RAW_DATA_DIR / "phishIRIS_DL_Dataset"
    SAME_TARGET: dict[(str, str)] = Field(
        default_factory=lambda: {
            "Absa Group": "absa",
            "Amazon.com Inc.": "amazon",
            "Americanas.com S,A Comercio Electrnico": "americanas",
            "Americanas.com S_A Comercio Electrnico": "americanas",
            "Aruba S.p.A.": "aruba",
            "Barclays Bank Plc": "barclays",
            "Canada Revenue Agency": "canada",
            "Capital One Financial Corporation": "capital_one",
            "Capitec Bank Limited": "capitec",
            "Chase Personal Banking": "chase",
            "Commonwealth Bank of Australia": "commonwealth_bank",
            "Cox Communications": "cox",
            "DHL Airways, Inc.": "dhl",
            "grupo_bancolombia": "bancolombia",
            "Bancolombia": "bancolombia",
            "HSBC Bank": "hsbc",
            "Itau Unibanco S.A": "itau",
            "La Banque postale": "la_banque_postale",
            "Lloyds TSB Group": "lloyds",
            "lloyds_bank": "lloyds",
            "Made-In-China": "made_in_china",
            "Microsoft OneDrive": "onedrive",
            "ms_onedrive": "onedrive",
            "ms_bing": "bing",
            "ms_office": "ms_office",
            "Office365": "ms_office",
            "National Australia Bank Limited": "nab",
            "NatWest Personal Banking": "natwest",
            "Navy Federal Credit Union": "navy_federal",
            "NedBank Limited": "nedbank",
            "OurTime Dating": "ourtime",
            "Outlook": "outlook",
            "ms_outlook": "outlook",
            "RBC Royal Bank": "rbc",
            "Sparkasse Bank": "sparkasse",
            "Standard Chartered Bank": "standard_bank",
            "Strato AG": "strato",
            "SunTrust Bank": "suntrust",
            "Swisscom IT Services AG": "swisscom",
            "Telecom Italia": "telecom",
            "Wells Fargo & Company": "wells_fargo",
        }
    )


config = AppConfig()
