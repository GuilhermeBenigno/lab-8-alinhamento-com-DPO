# Alinhamento de LLM com DPO

Projeto do **Laboratório 08** com foco em alinhar um modelo de linguagem utilizando **Direct Preference Optimization (DPO)**.

---

##  Objetivo

Treinar um modelo para preferir respostas mais seguras e adequadas com base em pares de preferência:

* `chosen` → resposta correta
* `rejected` → resposta inadequada

Seguindo o princípio HHH:

* Helpful
* Honest
* Harmless

---

##  Modelo

* Modelo base: Phi-2
* Quantização: 4-bit (QLoRA)
* Fine-tuning: LoRA + DPO

---

##  Dataset

Formato utilizado:

```json
{
  "prompt": "Pergunta",
  "chosen": "Resposta segura",
  "rejected": "Resposta incorreta"
}
```

Dataset com 30+ exemplos.

---

##  Treinamento

* Trainer: DPOTrainer
* Beta: 0.1
* Batch size: 1
* Precisão: float16
* Otimizações:

  * Gradient Checkpointing
  * Desativação de cache
  * Redução de dataset (para GPU limitada)

---

##  Explicação do Beta

O parâmetro **beta** controla o quanto o modelo pode se afastar do comportamento original.

* Beta baixo → mais adaptação
* Beta alto → mais conservador

Funciona como uma penalização (KL Divergence), mantendo equilíbrio entre aprendizado e estabilidade.

---

##  Estrutura

```
data/
src/
```

---

## 🚀 Como executar

### 1. Instalar dependências

```
pip install -r requirements.txt
```

### 2. Gerar dataset

```
python data/generate_dataset.py
```

### 3. Treinar modelo

```
python src/train_dpo.py
```

---

##  Resultado

O modelo treinado é salvo em:

```
dpo-model/
```

---

##  Observações

* Uso de quantização + LoRA foi necessário para rodar em GPU limitada (Google Colab)
* O DPO utiliza dois modelos (principal + referência), aumentando o consumo de memória
* Avisos de autenticação da Hugging Face não afetam a execução

---

##  Uso de IA

Partes geradas/complementadas com IA, revisadas por Guilherme Gomes 
