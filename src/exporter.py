import json
import uuid

class Exporter:
    def export(self, catalog, filename="catalog_seed.json"):
        with open(filename, "w") as f:
            json.dump(catalog, f, indent=2)