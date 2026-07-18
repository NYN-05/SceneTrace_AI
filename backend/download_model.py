"""Download Florence-2 model with unlimited timeout."""
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "999999"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "999999"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from transformers import AutoModelForCausalLM, AutoProcessor

model_name = "microsoft/Florence-2-base"
print(f"Downloading {model_name}...")
AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print("Done. Florence-2 model downloaded.")
