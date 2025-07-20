import pickle
from pathlib import Path


def prepare_domain_mappings():
    """Load domain map, add new mappings, and save back to file."""
    models_dir = Path(__file__).parent.parent / "models"
    domain_map_path = models_dir / "domain_map.pkl"

    # Load existing domain map
    with open(domain_map_path, "rb") as handle:
        domain_map = pickle.load(handle)

    # Add new domain mappings
    additional_mappings = {
        "Simplii Financial": ["simplii.com"],
        "Bank of The Bahamas Limited": ["bankbahamas.com"],
        "mdpd": ["mps.it"],
        "sicil_shop": ["sicilshop.com"],
    }

    # Update domain map with additional mappings
    domain_map.update(additional_mappings)

    # Save updated domain map
    with open(domain_map_path, "wb") as handle:
        pickle.dump(domain_map, handle)

    print(
        f"Domain mappings updated successfully. Added {len(additional_mappings)} new mappings."
    )


if __name__ == "__main__":
    prepare_domain_mappings()
