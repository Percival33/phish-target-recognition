from tools.ModelServing import ModelServing
from phishpedia import PhishpediaWrapper


class PhishpediaServing(ModelServing):
    def __init__(self):
        super().__init__()

    async def predict(self, data: dict):
        url = data.get("url", None)
        img = data.get("image", None)

        print(f"URL: {url}")
        print(f"type img: {type(img)}")

        (
            phish_category,
            pred_target,
            matched_domain,
            plotvis,
            siamese_conf,
            _,
            logo_recog_time,
            logo_match_time,
        ) = self.phishpedia.test_orig_phishpedia(url, None, None, img, None)

        return {
            "url": str(url),
            "class": int(phish_category),
            "target": str(pred_target),
            "confidence": float(siamese_conf),
        }

    async def on_startup(self):
        """Startup logic (e.g., loading resources)"""
        self.phishpedia = PhishpediaWrapper()


serving = PhishpediaServing()
app = serving.app

if __name__ == "__main__":
    serving.run()

# in api folder
# uv run python -m phishpedia.PhishpediaServing
