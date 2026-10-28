# DecisionLFM

Modelo de decisão próprio inspirado no CLEF, com alternativas descritas na
entrada e probabilidades calculadas em uma passagem pelo backbone.

## Avaliação inicial do LFM3-VL-2B

Avaliação documental e leitura do código oficial em **2026-10-28**.
**Recomendação: usar o modelo como candidato ao primeiro protótipo.**
Este documento registra a avaliação anterior à implementação. Consulte o
[README](README.md) para o código, a execução e os testes posteriores na RTX 4090.

Checkpoint: [`LiquidAI/LFM3-VL-2B`](https://huggingface.co/LiquidAI/LFM3-VL-2B).
Revisão consultada: `f83d2b0a2296aaa277f394990a9b8b949ddf9691`.

| Característica verificada | Consequência para o DecisionLFM |
| --- | --- |
| 1.918.181.888 parâmetros BF16 no Hub; aproximadamente 1,73B no módulo de linguagem | Candidato menor para experimentos na 4090 |
| Estados de linguagem com dimensão 2.048 | Nossa cabeça deve aceitar essa dimensão |
| 24 camadas: 13 convoluções, 5 atenções locais e 6 globais | Arquitetura híbrida; adaptar a integração ao LFM3 |
| `Lfm3VlModel.forward()` retorna `last_hidden_state` | Podemos pontuar alternativas sem gerar tokens |
| Texto sem imagens é aceito pelo `forward` | Começar pelo BANKING77; visão pode ser explorada depois |
| Contexto anunciado de 128K e português entre os idiomas | Recursos disponíveis; limite prático deve ser medido no nosso hardware |
| Código com gradient checkpointing e exemplo oficial com LoRA | Há um caminho documentado de adaptação; nosso treinamento usará perda de decisão |

O código de convolução usa o caminho PyTorch quando gradientes estão
habilitados, em vez dos kernels Triton destinados à inferência. Isso favorece
a adaptação, mas não substitui um teste real de `forward` e `backward`.

## Quatro componentes do primeiro protótipo

| Componente essencial | Proposta inicial |
| --- | --- |
| Cabeça que pontue alternativas | Pontuador compartilhado que recebe representações da pergunta/contexto e de cada alternativa; softmax entre alternativas válidas |
| Decisão em uma passagem | Uma chamada ao backbone para situação, pergunta e alternativas; leitura dos estados finais e aplicação da cabeça, sem `generate()` |
| Codificação da entrada | Formato textual explícito e estável, usando o tokenizer do LFM3; registrar posições das partes necessárias à cabeça |
| Treinamento para decisões | Entropia cruzada sobre a alternativa correta; primeiro cabeça com backbone congelado, depois cabeça + LoRA |

O pontuador deve aceitar quantidades e descrições variáveis de alternativas.
Uma camada com 77 saídas fixas resolveria apenas o classificador BANKING77 e
não demonstraria a capacidade de seguir novos esquemas de decisão.

Como o backbone é causal, a posição de leitura da decisão precisa ter acesso
à situação, à pergunta e às alternativas. O esquema de posições e a agregação
dos estados serão definidos na implementação. Verificar também sensibilidade
à ordem das alternativas e correção das máscaras de padding.

A cabeça será inicializada e treinada para nosso backbone. O nome ou o código
`JointSchemaHead` não são requisitos: o requisito é sua função de pontuar
alternativas a partir dos estados internos. Também não há motivo para repetir
automaticamente a remoção de MTP feita em outro checkpoint.

## Hardware e ambiente

- **Inferência BF16:** os pesos completos equivalem a aproximadamente 3,84 GB
  decimais (3,57 GiB). Ativações, buffers e contexto aumentam o consumo.
- **Treinar somente a cabeça:** caminho inicial de menor consumo, com backbone
  congelado; útil para validar a integração e obter uma primeira referência.
- **Cabeça + LoRA:** opção recomendada para adaptar o backbone na RTX 4090 de
  24 GB, começando com batch 1 e entradas curtas. A viabilidade e o pico real
  de memória precisam ser confirmados experimentalmente.
- **Todos os parâmetros:** não assumir que cabem em 24 GB; gradientes, estados
  do otimizador e ativações podem ultrapassar essa capacidade.

Usar ambiente separado dentro desta pasta para preservar o baseline CLEF.
O model card pede `transformers>=5.21.0` e `trust_remote_code=True`; o notebook
oficial de SFT fixa `transformers==5.20.2`. Essa diferença precisa ser resolvida
no primeiro teste, fixando as versões efetivamente validadas. O baseline usa
`transformers==5.10.2` e não deve ser atualizado para este experimento.
Fixar também a revisão do checkpoint e do código carregado.

O exemplo oficial demonstra SFT generativo com LoRA. Ele confirma um caminho
de ajuste do backbone, mas não implementa nossa cabeça nem a perda sobre as
alternativas. Não precisamos começar com um pipeline TRL completo.

## Referência adicional: d1-2B

A própria LiquidAI publica o [`d1-2B`](https://huggingface.co/LiquidAI/d1-2B),
um modelo de decisão treinado a partir do **LFM3-VL-2B**, com respostas em uma
passagem sem geração de tokens. É uma evidência concreta da adequação desse
backbone à tarefa e uma referência adicional para comparação futura.

Nosso DecisionLFM continuará sendo uma implementação e um treinamento próprios.
Somente adicionar uma cabeça ao LFM3 não demonstra novidade arquitetural, nem
garante o desempenho ou a calibração dos modelos já treinados para decisões.

## Licença

Os pesos usam a **LFM Open License v1.0**, com condições para uso e derivados.
Ela permite adaptação e redistribuição sob suas condições, incluindo preservação
da licença e dos avisos. Há limitação comercial associada ao patamar de receita
anual de US$ 10 milhões; a cláusula 5 prevê exceção para organizações sem fins
lucrativos qualificadas em uso não comercial ou de pesquisa. Consultar o
[texto oficial](https://www.liquid.ai/lfm-license) antes de distribuir pesos ou
planejar uso comercial. Não tratar os pesos como licenciados em Apache 2.0.

## Sequência proposta após esta avaliação

1. Criar ambiente isolado e validar o carregamento da revisão fixada.
2. Executar uma entrada textual, verificar formato e finitude dos estados
   finais e confirmar gradientes; medir tempo e VRAM.
3. Implementar codificador e cabeça simples para uma pergunta de escolha única.
4. Treinar usando somente o `train` do BANKING77, reservando validação dentro
   dele. Usar o `test` oficial apenas após congelar as escolhas do experimento.
5. Avaliar com as métricas e o formato de saídas do baseline: acurácia,
   macro-F1, Brier, latência, memória e falhas.

O CLEF-Flash permanece como baseline **zero-shot**. Um DecisionLFM treinado no
BANKING77 deve ser identificado como **supervisionado nesse dataset**. Comparar
os resultados sob essa distinção; testar outros datasets e esquemas ainda será
necessário para avaliar generalização. Probabilidades produzidas por softmax
também precisam de avaliação de calibração.

## Fontes consultadas

- [Model card na revisão avaliada](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/f83d2b0a2296aaa277f394990a9b8b949ddf9691/README.md)
- [Configuração do modelo](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/f83d2b0a2296aaa277f394990a9b8b949ddf9691/config.json)
- [Backbone multimodal e retorno dos estados](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/f83d2b0a2296aaa277f394990a9b8b949ddf9691/modeling_lfm3_vl.py)
- [Backbone de linguagem](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/f83d2b0a2296aaa277f394990a9b8b949ddf9691/modeling_lfm3.py)
- [Caminho de convolução com e sem gradientes](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/f83d2b0a2296aaa277f394990a9b8b949ddf9691/lfm3_conv.py)
- [Notebook oficial de SFT + LoRA](https://github.com/Liquid4All/cookbook/blob/main/finetuning/notebooks/lfm3_vl_sft_with_trl.ipynb)
- [d1-2B, decisão sobre o mesmo backbone](https://huggingface.co/LiquidAI/d1-2B)
- [Licença do checkpoint](https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/f83d2b0a2296aaa277f394990a9b8b949ddf9691/LICENSE)
