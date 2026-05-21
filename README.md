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