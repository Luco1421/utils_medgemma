"""Non-contractual MedGemma-LoRA extension kept outside public M7."""

from __future__ import annotations

from typing import Any

from .medgemma_conditioner import MedGemmaConditioner


class ExperimentalLoRAMedGemmaConditioner(MedGemmaConditioner):
    """Load an experimental PEFT adapter on top of the M7 base model."""

    def __init__(self, config: dict[str, Any]):
        adapter_path = config.get("adapter_path")
        if not adapter_path:
            raise ValueError("Experimental MedGemma-LoRA requires adapter_path")
        base_config = {
            key: value for key, value in config.items() if key != "adapter_path"
        }
        super().__init__(base_config)

        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.config["adapter_path"] = adapter_path
