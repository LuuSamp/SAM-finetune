# SAM Fine-tuning para Segmentação de Pólipos

Este projeto implementa o fine-tuning do **Segment Anything Model (SAM)** da Meta AI para segmentação de pólipos gastrointestinais utilizando o dataset **Kvasir-SEG**. O modelo é adaptado especificamente para tarefas de segmentação médica, alcançando melhorias significativas em relação ao modelo pré-treinado.

## Índice

- [Estrutura do Projeto](#estrutura-do-projeto)
- [Sobre o Projeto](#sobre-o-projeto)
- [Características Principais](#características-principais)
- [Resultados](#resultados)
- [Instalação](#instalação)
- [Uso](#uso)
- [Configurações](#configurações)
- [Estrutura de Arquivos](#estrutura-de-arquivos)
- [Apresentação](#apresentação-do-trabalho)

## Estrutura do Projeto

```
SAM-finetune/
│
├── model/                      # Módulos principais do projeto
│   ├── __init__.py
│   ├── config.py              # Configurações do modelo
│   ├── dataloaders.py         # Carregamento e preparação de dados
│   ├── evaluation.py          # Funções de avaliação e visualização
│   ├── loss.py                # Funções de perda combinadas
│   ├── sam_model.py           # Classe principal para fine-tuning
│   └── setup.py               # Setup automático do ambiente
│
├── outputs/                    # Resultados do treinamento
│   ├── best_model.pth         # Melhor modelo salvo
│   ├── checkpoint_epoch_*.pth  # Checkpoints periódicos
│   ├── training_curves.png     # Gráficos de treinamento
│   ├── predictions.png         # Visualizações de predições
│   └── results.json            # Métricas salvas
│
├── main.ipynb                  # Notebook principal com análises
├── train.py                    # Script de treinamento
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

## Sobre o Projeto

O **Segment Anything Model (SAM)** é um modelo de segmentação de última geração desenvolvido pela Meta AI, capaz de segmentar qualquer objeto em uma imagem com base em prompts interativos (pontos, caixas delimitadoras ou máscaras). Este projeto adapta o SAM para a tarefa específica de segmentação de pólipos gastrointestinais através de fine-tuning no dataset Kvasir-SEG.

### Objetivos

- Fine-tuning do SAM para segmentação de pólipos gastrointestinais
- Comparação entre modelo pré-treinado e modelo fine-tuned
- Avaliação de diferentes estratégias de prompts (pontos e bounding boxes)
- Análise detalhada de performance em casos desafiadores

## Características Principais

- **Fine-tuning Eficiente**: Apenas o decoder de máscaras é treinado, mantendo o encoder de imagens e o encoder de prompts congelados
- **Múltiplos Tipos de Prompts**: Suporte para prompts de pontos, bounding boxes ou ambos simultaneamente
- **Data Augmentation**: Transformações de dados para melhorar a generalização
- **Avaliação Completa**: Métricas de IoU (Intersection over Union) e Dice Score
- **Visualizações**: Gráficos de comparação e exemplos visuais de predições
- **Setup Automatizado**: Download automático do dataset e checkpoint do SAM

## Resultados

O modelo fine-tuned demonstra melhorias significativas em relação ao modelo pré-treinado:

| Métrica | Modelo Base | Modelo Fine-tuned | Melhoria |
|---------|-------------|-------------------|----------|
| **Loss** | 0.1662 | 0.0630 | **-62.1%** |
| **IoU** | 0.7966 | 0.8896 | **+11.7%** |
| **Dice Score** | 0.8712 | 0.9367 | **+7.5%** |

### Estatísticas Adicionais

- **IoU Médio**: 0.8896 ± 0.1088 (vs 0.7966 ± 0.1860 no modelo base)
- **Melhoria Média**: +0.0930 em IoU
- **Casos Desafiadores**: Em casos difíceis (IoU < 0.7 no modelo base), o modelo fine-tuned alcança IoU médio de 0.7707 (vs 0.4726 no modelo base)

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/LuuSamp/SAM-finetune.git
cd SAM-finetune
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Setup Automático

O ambiente será configurado automaticamente na primeira execução, incluindo:
- Download do dataset Kvasir-SEG
- Download do checkpoint do SAM (vit_b)
- Instalação de dependências adicionais

## Uso

### Treinamento

Para treinar o modelo, execute:

```bash
python train.py
```

O script irá:
1. Configurar o ambiente automaticamente
2. Preparar os dados (split train/val/test)
3. Treinar o modelo por 60 épocas
4. Salvar o melhor modelo e checkpoints
5. Gerar visualizações e métricas

### Uso no Notebook

Abra o `main.ipynb` para:
- Visualizar exemplos dos dataloaders
- Comparar modelo base vs fine-tuned
- Analisar resultados detalhados
- Visualizar predições em múltiplas imagens

### Carregar Modelo Treinado

```python
from model.config import Config
from model.sam_model import SAMFineTuner

# Inicializar configuração
config = Config()

# Carregar modelo fine-tuned
model = SAMFineTuner(config).load('outputs/best_model.pth')

# Criar modelo original para comparação
model_original = SAMFineTuner.create_original(config)
```

## Configurações

As principais configurações podem ser ajustadas em `model/config.py`:

### Parâmetros de Treinamento

```python
BATCH_SIZE = 4              # Tamanho do batch
NUM_EPOCHS = 60             # Número de épocas
LEARNING_RATE = 1e-5        # Taxa de aprendizado
WEIGHT_DECAY = 1e-4         # Decaimento de pesos
```

### Divisão dos Dados

```python
TRAIN_SPLIT = 0.7           # 70% para treino
VAL_SPLIT = 0.15            # 15% para validação
TEST_SPLIT = 0.15           # 15% para teste
```

### Estratégia de Prompts

```python
USE_BOX_PROMPTS = True      # Habilitar bounding boxes
USE_BOTH_PROMPTS = True     # Usar pontos e boxes simultaneamente
PROMPT_MIX_RATIO = 0.5      # Razão boxes vs pontos (quando não usar ambos)
```

### Congelamento de Camadas

```python
FREEZE_IMAGE_ENCODER = True  # Congelar encoder de imagens
FREEZE_PROMPT_ENCODER = True # Congelar encoder de prompts
# Apenas o mask_decoder é treinado (4.33% dos parâmetros)
```

### Processamento de Imagens

```python
IMAGE_SIZE = 1024           # Tamanho das imagens (requerido pelo SAM)
USE_AUGMENTATION = True     # Habilitar data augmentation
```

## Estrutura de Arquivos 

### `model/config.py`
Classe de configuração centralizada que gerencia todos os parâmetros do projeto e inicializa o ambiente automaticamente.

### `model/sam_model.py`
Implementação principal do fine-tuning:
- `SAMFineTuner`: Classe principal para treinamento e inferência
- `train_epoch()`: Treina por uma época
- `validate()`: Avalia no conjunto de validação/teste
- `fit()`: Loop principal de treinamento

### `model/dataloaders.py`
- `prepare_data_splits()`: Divide o dataset em train/val/test
- `create_data_loaders()`: Cria dataloaders com augmentation

### `model/evaluation.py`
- `calculate_iou()`: Calcula Intersection over Union
- `calculate_dice()`: Calcula Dice Score
- `compare_models()`: Compara dois modelos
- `plot_training_curves()`: Gera gráficos de treinamento
- `visualize_predictions()`: Visualiza predições

### `model/loss.py`
Implementa `CombinedLoss`, combinando Binary Cross-Entropy e Dice Loss para melhor treinamento.

### `model/setup.py`
Funções utilitárias para:
- Download automático do dataset Kvasir-SEG
- Download do checkpoint do SAM
- Instalação de dependências

## Métricas de Avaliação

O projeto utiliza três métricas principais:

1. **Loss**: Perda combinada (BCE + Dice Loss)
2. **IoU (Intersection over Union)**: Mede a sobreposição entre predição e ground truth
3. **Dice Score**: Mede a similaridade entre predição e ground truth

## Dataset

O projeto utiliza o **Kvasir-SEG**, um dataset público para segmentação de pólipos gastrointestinais:
- **1000 imagens** com máscaras correspondentes
- Imagens de endoscopia gastrointestinal
- Máscaras binárias para segmentação de pólipos

O dataset é baixado automaticamente na primeira execução.

## Arquitetura do Modelo

O SAM (Segment Anything Model) consiste em três componentes principais:

1. **Image Encoder**: Vision Transformer (ViT) que processa imagens
2. **Prompt Encoder**: Processa prompts (pontos, boxes, máscaras)
3. **Mask Decoder**: Gera máscaras de segmentação

**Estratégia de Fine-tuning**: Apenas o **Mask Decoder** é treinado (4.33% dos parâmetros), mantendo os encoders congelados para eficiência e prevenção de overfitting.

---
## Apresentação do trabalho
[SAM - Apresentação](https://www.youtube.com/watch?v=Vrisz5Q8Gfk) (link: [https://www.youtube.com/watch?v=Vrisz5Q8Gfk](https://www.youtube.com/watch?v=Vrisz5Q8Gfk))
