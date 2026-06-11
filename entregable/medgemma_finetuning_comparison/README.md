# MedGemma Finetuning Comparison

Experimento reproducible para comparar MedGemma base contra MedGemma fine-tuned con
LoRA/QLoRA en descripciones oftalmologicas.

Este entregable no reemplaza los notebooks de pruebas. Los notebooks siguen siendo
el espacio de depuracion en Kabre; esta carpeta contiene la version modular en
formato de proyecto.

## Estructura

```text
medgemma_finetuning_comparison/
  config.yaml
  SPEC.md
  modules/
    data_module.py
    evaluator.py
    lora_trainer.py
    medgemma_model.py
    utils.py
  scripts/
    evaluate_base.py
    train_lora.py
    evaluate_lora.py
    run_comparison.py
  slurm/
    run_comparison.slurm
  results/
  checkpoints/
```

## Ejecucion local o en cluster

Desde esta carpeta:

```bash
export HF_TOKEN=...
python scripts/run_comparison.py --config config.yaml
```

En Kabre:

```bash
export HF_TOKEN=...
sbatch slurm/run_comparison.slurm
```

## Salidas

- Adapter LoRA: `checkpoints/official_medgemma_qlora_description/`
- Resultados base: `results/base_results.json`
- Resultados LoRA: `results/lora_results.json`
- Comparacion final: `results/comparison.json`

El adapter guardado no es una copia completa de MedGemma. Para inferencia se carga
el modelo base de Hugging Face y luego el adapter LoRA.
