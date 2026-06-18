**Em Construção**

# Projeto de Computação Evolucionária

Este projeto visa replicar o artigo **A genetic algorithm-based ensemble framework for wind speed forecasting**, a fim de atender as necessidades da disciplina de Computação Evolucionária, semestre 01/2026.


## Conceitos e Passos do GA utilizados

1. Inicialização
    Criação aleatória de um conjunto de soluções candidatas (população inicial).
    
    Aqui são criados individuos de forma aleatória para cada um dos modelos que serão utilizados.

2. Avaliação

Aplica-se uma função de fitness no conjunto.
Essa função define o quão boa é cada solução (quanto maior ou menor, melhor — depende do problema).
* Nesse processo, os individuos inicializados serão executados em cada modelo e avaliados de acordo com o resultado do fitness utilizado.

3. Seleção
Alguns dos melhores candidatos são escolhidos.
Esses candidatos irão gerar a próxima geração.

4. Operadores de variação

4.1 Cruzamento (Recombinação)
A partir de dois ou mais candidatos (pais), são gerados novos indivíduos (filhos).

4.2 Mutação
Pequenas alterações aleatórias são aplicadas nos filhos, criando descendentes. (offspring)
Isso cria diversidade nas soluções (novos descendentes).

5. Seleção da nova geração
Forma-se um novo conjunto de soluções.
Pode incluir:
Apenas os filhos, ou
Filhos + alguns pais (estratégia elitista).

6. Critério de parada
O processo é repetido até que uma condição seja atendida, como:
Número máximo de gerações
Solução suficientemente boa encontrada

### Professor: Domingos Sávio
### Aluno: Thale Mendes

# Conteúdo

**Algoritmo Genético**
```Notebook 1
em construção
```

**Modelos**
```Notebook 2
em construção
```

**Pré-processamento/Normalização**
```Notebook 3
em construção
```


## Infos Gerais

<!-- # GA para SVR - No SVR não há pesos para otimizar, mas pode atuar diretamente nos hiperparâmetros (C, epsilon, gamma) para melhorar a performance. O processo é similar ao do MLP, mas o cromossomo representa uma combinação de hiperparâmetros em vez de pesos da rede. O GA busca a melhor combinação que minimize o erro de previsão no conjunto de validação.

# cromossomo = [C, epsilon, gamma]
# fitness    = -RMSE no conjunto de validação
# GA busca a melhor combinação desses três valores

# Sem GA — busca manual 
# for c in [0.1, 1, 10, 100]:
#     for e in [0.01, 0.05, 0.1]:
#         for g in ['scale', 0.01, 0.1]:
#             ...  # 36 combinações fixas

# Com GA — busca contínua e inteligente
# cromossomo = [C=7.3, epsilon=0.032, gamma=0.07]
# GA explora o espaço contínuo, não uma grade fixa -->

# # Hiperparâmetros do SVR
# C — penalidade por erro:

# Controla o quanto o modelo penaliza pontos que ficam fora do tubo ε
# C pequeno → modelo mais suave, aceita mais erros, pode underfit
# C grande → modelo mais rígido, tenta acertar tudo, pode overfit
# Valores típicos para começar: [0.1, 1, 10, 100]

# epsilon — largura do tubo de tolerância:

# Pontos dentro do tubo não contribuem para o erro — são ignorados
# Epsilon pequeno → tubo estreito, modelo mais sensível a cada ponto
# Epsilon grande → tubo largo, modelo mais suave
# Valores típicos: [0.01, 0.05, 0.1, 0.5]
# Regra prática: comece com 5-10% do desvio padrão do y de treino

# gamma — curvatura do kernel RBF:

# Controla o quanto cada ponto de treino influencia a curva
# 'scale' → 1 / (n_features * X.var()) — automático
# Gamma pequeno → curva mais suave, influência mais ampla
# Gamma grande → curva mais irregular, influência localizada
# Valores típicos: ['scale', 0.001, 0.01, 0.1, 1]