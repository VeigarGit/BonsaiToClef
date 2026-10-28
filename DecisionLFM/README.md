# DecisionLFM

Modelo de decisão textual baseado no **LiquidAI/LFM3-VL-2B**. Recebe uma
situação, uma pergunta e alternativas descritas em texto; devolve probabilidades
em uma passagem, sem gerar tokens. A [avaliação do backbone](AVALIACAO.md)
registra a escolha inicial e as fontes.

## Arquivos

- `modelo.py`: codificação, cabeça, LoRA, salvamento e recarregamento.
- `dados.py`: revisão fixa do BANKING77 e critérios iguais aos do CLEF.
- `treinar.py`: treino supervisionado e seleção por validação.
- `inferir.py`: executa o modelo salvo e escreve as previsões.
- `verificar.py`: recarrega o checkpoint e reproduz sua validação.
- `comparar.py`: compara resultados, recusando protocolos ou amostras diferentes.
- `test_modelo.py` e `test_comparar.py`: verificações de alinhamento, padding,
  alternativas variáveis, gradientes da cabeça e comparação.

A avaliação usa diretamente o **`avaliar.py` na raiz do repositório**.
O código e o ambiente do CLEF permanecem independentes.

## 1. Preparar o ambiente no avancos

Requer GPU CUDA com suporte a BF16; a implementação inicial foi preparada para
a RTX 4090 de 24 GB. Não usa quantização, imagens ou vídeo nesta etapa.

```bash
ssh avancos
cd ~/BonsaiToClef/DecisionLFM
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Todos os comandos seguintes partem de `~/BonsaiToClef/DecisionLFM`.
Não instalar estas dependências na `.venv` do CLEF, na pasta pai.
Usamos Transformers 5.21.0, PEFT 0.21.2 e PyTorch 2.11.0; o modelo e os dados
são públicos e carregados sem depender de token de acesso. O primeiro uso baixa
os pesos completos para o cache do Hugging Face.

O código oficial é carregado com `trust_remote_code=True`, na revisão fixa
`f83d2b0a2296aaa277f394990a9b8b949ddf9691`. Os pesos usam a
[LFM Open License](https://www.liquid.ai/lfm-license), com condições próprias
para uso comercial e redistribuição.

## 2. Conferir os testes pequenos

```bash
.venv/bin/python -m unittest -v test_modelo test_comparar
```

Esses testes usam o tokenizer oficial e uma cabeça pequena, sem carregar o
backbone completo. Não substituem os testes de treinamento e recarregamento.

## 3. Fazer um treino rápido de integração

```bash
.venv/bin/python treinar.py \
  --output-dir modelos/smoke-novo \
  --train-limit 32 --validation-limit 16 \
  --head-epochs 1 --lora-epochs 1 \
  --batch-size 4 --gradient-accumulation 2 --max-steps 2

.venv/bin/python inferir.py \
  --checkpoint modelos/smoke-novo \
  --split train --limit 20 --output-dir resultados/smoke-novo

.venv/bin/python ../avaliar.py resultados/smoke-novo
```

Este teste confirma o percurso treino → salvamento → recarregamento → avaliação.
Os números do smoke não representam desempenho final; os 20 exemplos de `train`
podem sobrepor o treino ou a validação. Usar diretórios novos em cada execução.
Diretórios de treino e de inferência existentes são recusados.

## 4. Treinar o modelo completo

```bash
.venv/bin/python -u treinar.py --output-dir modelos/banking77-novo
```

O padrão faz uma época de cabeça com backbone congelado e três épocas de
cabeça + LoRA. Batch 4 com acumulação de 4 corresponde a até 16 exemplos por
atualização. A cabeça tem dimensão intermediária 256; LoRA usa rank 8 nas
projeções de atenção, convolução e MLP do módulo de linguagem.

O script carrega **apenas `train`**, separa 10% para validação estratificada
com seed 42 e registra os índices. Não usa o `test` para selecionar pesos.
Todas as 77 alternativas participam de cada exemplo. Sua ordem é embaralhada
uma vez por exemplo de treino, de forma reproduzível; a validação usa a ordem
canônica. A loss é entropia cruzada sobre a alternativa correta.

A situação vem depois das alternativas no prompt. O estado final da sequência
vê situação, pergunta e todas as alternativas. A cabeça compara esse estado
com a média dos estados dos tokens de cada alternativa, usando projeções
treináveis e similaridade. A quantidade de saídas depende das alternativas
recebidas, sem uma camada fixa de 77 classes.

Na primeira fase, salvamos a melhor cabeça por acurácia de validação. Ela
inicializa a fase LoRA; o checkpoint final é o de maior acurácia de validação
**dentro da fase LoRA**, mantendo o primeiro em caso de empate. Com
`--lora-epochs 0`, o resultado contém somente a cabeça treinada.

Os principais argumentos são `--batch-size`, `--gradient-accumulation`,
`--head-epochs`, `--lora-epochs`, `--head-lr`, `--lora-lr` e `--max-length`.
O limite padrão é 8.192 tokens. Entradas maiores são recusadas, sem truncamento.

## 5. O que fica salvo

```text
modelos/banking77-novo/
  decision_config.json     # revisão do backbone, cabeça e formato
  head.safetensors        # cabeça treinada
  adapter/                # pesos e configuração LoRA, quando habilitado
  tokenizer/              # tokenizer usado no experimento
  training.json           # índices, parâmetros, versões, hashes e validação
```

Esse conjunto é o **checkpoint do DecisionLFM**. O backbone congelado não é
duplicado: o carregador recupera sua revisão fixa do cache ou do Hugging Face
e aplica os adaptadores salvos. Copiar a pasta inteira ao mover o checkpoint;
em outra máquina também será necessário obter o backbone.

`training.json` registra o histórico, a época selecionada, hashes dos scripts
presentes no início do treino, versões, GPU, tempo e pico de memória alocada. A inferência
de benchmark recusa treinamento marcado como incompleto. Pesos e resultados
ficam ignorados pelo Git em `modelos/` e `resultados/`.

Para conferir o checkpoint salvo antes do teste final:

```bash
.venv/bin/python verificar.py --checkpoint modelos/banking77-novo
```

O verificador exige que a acurácia e a loss de validação sejam preservadas ao
recarregar, confirma uma chamada ao backbone por decisão e testa a interface
com 77 e duas alternativas. Ele usa a validação, sem consultar `test`.

## 6. Executar o mesmo teste do CLEF

Depois de concluir treino e validação, usar o checkpoint escolhido:

```bash
.venv/bin/python inferir.py \
  --checkpoint modelos/banking77-novo \
  --split test --output-dir resultados/test-novo

.venv/bin/python ../avaliar.py resultados/test-novo

.venv/bin/python comparar.py ../resultados/test resultados/test-novo
```

O primeiro diretório do comparador é o resultado CLEF existente na raiz. São
os mesmos 3.080 textos do teste oficial, as mesmas 77 alternativas e descrições,
as mesmas instruções, batch 1 e três chamadas de aquecimento. O dataset está
fixado em `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`.

O script salva `run.json` e `predictions.jsonl`, aceitos pelo avaliador original,
que gera `metrics.json`: acurácia, macro-F1, Brier, latência mediana/p95, VRAM e
falhas. `comparar.py` confere IDs, textos, rótulos e parâmetros do protocolo,
e salva `comparison.json`. As inferências são sequenciais, sem compartilhar
a GPU com o treinamento.

Latência inclui codificação, transferência, inferência e obtenção das
probabilidades com sincronização CUDA, excluindo carregamento, aquecimento e
escrita. Memória é `max_memory_allocated`, incluindo pesos. O prompt e a
codificação interna diferem entre os modelos; o número de tokens também pode
ser diferente. A cabeça usa FP32 e o backbone BF16.

**CLEF é zero-shot; DecisionLFM é supervisionado no BANKING77.** A comparação
mostra desempenho e custo nesse cenário, sem estabelecer superioridade
zero-shot ou generalização para novos datasets. Softmax não garante calibração;
o Brier ajuda a avaliar as probabilidades.

## 7. Usar o checkpoint em código

```python
from modelo import DecisionLFM

model = DecisionLFM.load("modelos/banking77-novo")
result = model.decidir(
    "My card payment was declined.",
    "Which banking intent best describes this message?",
    {"card_payment": "Card payment declined", "cash": "Cash withdrawal problem"},
)
print(result)
```

O retorno contém `prediction`, `confidence`, `probabilities` e `input_tokens`.
Aceitar novas alternativas é uma propriedade da interface; a qualidade em
novas descrições, idiomas e domínios precisa de avaliação própria.

## 8. Execução validada em 2026-10-28

O checkpoint pronto está em **`modelos/banking77`**, no `avancos`, com cópia
local. Treinamos com os parâmetros padrão: 9.002 exemplos para treino e 1.001
para validação, uma época de cabeça e três de cabeça + LoRA. A última época
foi selecionada, com 91,61% de acurácia na validação. O ciclo de treino,
incluindo validações e salvamentos, levou 36,48 minutos e atingiu 4,48 GiB de
VRAM alocada. Cabeça: 1.052.673 parâmetros; cabeça + LoRA: 9.105.409 treináveis.

O checkpoint foi recarregado **offline** e reproduziu exatamente a acurácia e
a loss dos 1.001 exemplos de validação. O verificador observou uma chamada ao
backbone por decisão e confirmou a interface com 77 e duas alternativas. Os
cinco testes automatizados também passaram.

Em seguida, avaliamos todos os 3.080 exemplos do teste oficial, com 40 exemplos
por classe, sem alterações no modelo após consultar o teste. As previsões e
os metadados estão em `resultados/test/`; o avaliador da raiz calculou:

| Métrica | CLEF-Flash zero-shot | DecisionLFM supervisionado |
| --- | ---: | ---: |
| Acurácia | 95,91% | 92,79% |
| Macro-F1 | 95,90% | 92,75% |
| Brier multiclasses | 0,067443 | 0,114599 |
| Latência mediana | 312,77 ms | 24,18 ms |
| Latência p95 | 313,67 ms | 24,65 ms |
| Pico de VRAM alocada | 18,16 GiB | 3,68 GiB |
| Falhas | 0 | 0 |

Esses valores comparam os pipelines executados na mesma RTX 4090. O CLEF
teve melhor acurácia e Brier; o DecisionLFM usou menos memória e apresentou
menor latência neste protocolo. As entradas codificadas tiveram 455–531 tokens
no DecisionLFM e 1.826–1.902 no CLEF, além das diferenças de tamanho do modelo,
treinamento e software. Os números não isolam o efeito da arquitetura.

Para conferir o modelo já treinado e repetir a avaliação em outro diretório:

```bash
.venv/bin/python verificar.py --checkpoint modelos/banking77
.venv/bin/python inferir.py \
  --checkpoint modelos/banking77 --split test --output-dir resultados/repeticao
.venv/bin/python ../avaliar.py resultados/repeticao
.venv/bin/python comparar.py ../resultados/test resultados/repeticao
```

`resultados/test/comparison.json` contém a comparação completa e
`modelos/banking77/training.json` identifica o checkpoint selecionado. A pasta
do checkpoint validado também inclui `environment.txt` com o ambiente instalado
e `BASE_MODEL_LICENSE.txt` com a licença original dos pesos do backbone.
