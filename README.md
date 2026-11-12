# Trabalho: implementação manual do K-means

**Matemática Computacional** · Prof. Dr. Alexsandro O. A.

Clusterização com vetores e matrizes, do zero, em Python.

## Sumário

1. [Objetivo e regras](#objetivo-e-regras)
2. [Fórmulas](#fórmulas)
3. [Pseudocódigo e convenções](#pseudocódigo-e-convenções)
4. [Dataset 1: entregas dos Correios (resultados de teste)](#dataset-1-entregas-dos-correios-resultados-de-teste)
5. [Dataset 2: vendas de produtos esportivos (enunciado)](#dataset-2-vendas-de-produtos-esportivos-enunciado)
6. [Dataset 3: produtos bancários (enunciado)](#dataset-3-produtos-bancários-enunciado)
7. [Entrega e avaliação](#entrega-e-avaliação)

## Objetivo e regras

Neste trabalho você vai implementar o K-means do zero, em Python, usando apenas vetores e matrizes do NumPy. Depois de validar a implementação com os resultados de teste do dataset de entregas, você vai usá-la nos enunciados dos outros dois datasets. As perguntas de cada enunciado estão no sistema da disciplina.

Regras:

- O trabalho pode ser feito individualmente ou em dupla.
- Só podem ser usadas as bibliotecas NumPy, pandas e Matplotlib. É proibido usar `sklearn`, `scipy.cluster` ou qualquer outra implementação pronta de K-means ou de silhueta.
- A implementação deve seguir exatamente o pseudocódigo e as convenções deste enunciado. Só assim os resultados batem com os testes e com as respostas esperadas nos enunciados.

Arquivos fornecidos:

| Arquivo | Conteúdo |
| --- | --- |
| `kmeans_template.py` | Template com as funções a implementar e os testes automáticos do dataset 1 |
| `entregas_correios.csv` | Dataset 1: 60 entregas em uma cidade (coordenadas, peso, prazo, tipo de serviço) |
| `vendas_esportivas.csv` | Dataset 2: 120 clientes de uma loja de artigos esportivos |
| `clientes_banco.csv` | Dataset 3: 160 clientes de um banco |

Coloque os quatro arquivos na mesma pasta e rode `python kmeans_template.py`. Enquanto uma função não estiver pronta, o template avisa qual é.

## Fórmulas

Os dados formam a matriz $`X`$, com $`n`$ linhas (os dados $`x_1, \dots, x_n`$) e $`d`$ colunas (os atributos). Os $`k`$ centroides formam a matriz $`C`$, com $`k`$ linhas.

**Padronização (z-score).** Para cada coluna $`j`$, com média $`\mu_j`$ e desvio-padrão populacional $`\sigma_j`$ (divisão por $`n`$, `ddof=0`):

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

$$\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij} \qquad \sigma_j = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2}$$

**Distância ao quadrado** entre o dado $`x_i`$ e o centroide $`c_k`$, que forma a matriz $`D`$ ($`n \times k`$):

$$D_{ik} = \|x_i - c_k\|^2 = \sum_{j=1}^{d} (x_{ij} - c_{kj})^2$$

**Atribuição.** Cada dado vai para o centroide mais próximo:

$$r_i = \arg\min_{k} D_{ik}$$

**Atualização.** Cada centroide vira a média dos dados do seu grupo, em que $`|C_k|`$ é o número de dados do grupo $`k`$:

$$c_k = \frac{1}{|C_k|} \sum_{i : r_i = k} x_i$$

**Função objetivo (inércia, WCSS):**

$$J = \sum_{i=1}^{n} \|x_i - c_{r_i}\|^2$$

**Silhueta.** Para o dado $`i`$, do grupo $`A`$, com a distância euclidiana sem o quadrado. A distância média aos colegas de grupo é

$$a(i) = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \neq i} \|x_i - x_j\|$$

e a distância média ao grupo vizinho mais próximo é

$$b(i) = \min_{B \neq A} \frac{1}{|C_B|} \sum_{j \in C_B} \|x_i - x_j\|$$

O coeficiente de cada dado e a silhueta média são

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \qquad \bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

Se o grupo do dado $`i`$ tiver um único elemento, use $`s(i) = 0`$.

**Desfazer a padronização** de um centroide, para interpretá-lo nas unidades originais:

$$c_{kj}^{orig} = c_{kj} \cdot \sigma_j + \mu_j$$

## Pseudocódigo e convenções

```text
ALGORITMO K-MEANS(X, C_inicial, max_iter = 100)
    C ← C_inicial                      # k × d
    r_anterior ← NENHUM
    historico_J ← lista vazia
    n_atualizacoes ← 0

    PARA t = 1, 2, ..., max_iter FAÇA
        # Passo 1: atribuir
        D ← matriz n × k com D[i, k] = ||x_i − c_k||²
        r ← vetor n com r[i] = índice da menor entrada da linha i de D

        # Critério de parada
        SE r_anterior ≠ NENHUM E r = r_anterior ENTÃO
            PARAR

        # Passo 2: atualizar
        PARA cada grupo k FAÇA
            SE algum r[i] = k ENTÃO
                c_k ← média dos x_i com r[i] = k
            SENÃO
                c_k permanece igual        # grupo vazio

        n_atualizacoes ← n_atualizacoes + 1
        acrescentar J(X, C, r) a historico_J
        r_anterior ← r

    RETORNAR r, C, historico_J, n_atualizacoes
```

Convenções obrigatórias:

- **Centroides iniciais:** cada dataset tem uma lista de inicialização com ids (os valores da coluna de id do CSV, não a posição da linha). Para um dado k, pegue os k primeiros ids dessa lista; os centroides iniciais são as linhas dos dados (já padronizados, quando for o caso) que têm esses ids. O grupo 1 é o do primeiro id, o grupo 2 é o do segundo, e assim por diante. Exemplo: no dataset 1, `INIT_LOG = [2, 4, 6, 8, 10, 12]`; com k = 3, os 3 primeiros ids são 2, 4 e 6, então os centroides iniciais são as entregas de id 2, 4 e 6 (e não as três primeiras linhas do arquivo).
- **Empates:** se dois centroides estiverem à mesma distância, vence o de menor índice (é o que `np.argmin` faz).
- **Grupo vazio:** se nenhum dado for atribuído a um grupo, seu centroide não muda.
- **Contagem:** o número pedido nos testes e enunciados é o de atualizações dos centroides (`n_atualizacoes`). A última iteração, que só confirma que nada mudou, não conta.
- **J:** é calculado depois de cada atualização, com os centroides novos e as atribuições da mesma iteração. O J final é o último valor do histórico.
- **Padronização:** nos datasets 2 e 3, padronize as colunas indicadas antes de rodar o K-means e use a padronização também para inicializar. As colunas de texto e de identificação nunca entram em X.
- **Silhueta:** calcule sempre nos mesmos dados usados pelo K-means (padronizados, quando for o caso).

## Dataset 1: entregas dos Correios (resultados de teste)

Uma agência quer dividir as 60 entregas do dia entre 3 carteiros, agrupando-as por região da cidade. Use apenas as colunas `x_km` e `y_km` (posição em km a partir do centro), **sem padronizar**, pois as duas já estão na mesma unidade. Use k = 3 e a lista de inicialização `INIT_LOG = [2, 4, 6, 8, 10, 12]`; para k = 3, os centroides iniciais são as entregas cujos ids são os 3 primeiros da lista: 2, 4 e 6.

Compare cada etapa da sua implementação com os valores abaixo, arredondados para 2 casas. O template já faz essas comparações automaticamente.

**Teste 1: centroides iniciais e matriz D.** C inicial = (10,33; 14,50), (13,11; 6,39) e (7,46; 16,07). As cinco primeiras linhas de D devem ser:

| id | x_km | y_km | D[·,1] | D[·,2] | D[·,3] | Grupo |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15,93 | 5,17 | 118,41 | 9,44 | 190,55 | 2 |
| 2 | 10,33 | 14,50 | 0,00 | 73,50 | 10,70 | 1 |
| 3 | 3,05 | 5,29 | 137,82 | 102,41 | 135,66 | 2 |
| 4 | 13,11 | 6,39 | 73,50 | 0,00 | 125,62 | 2 |
| 5 | 12,17 | 6,25 | 71,45 | 0,90 | 118,62 | 2 |

**Teste 2: primeira iteração.** Depois da primeira atribuição, os grupos têm 8, 41 e 11 entregas. Depois da primeira atualização, os centroides são (9,71; 14,19), (8,99; 5,77) e (7,35; 14,67), e J = 1494,25.

**Teste 3: convergência.** O algoritmo faz 8 atualizações. O histórico de J é 1494,25 · 1406,25 · 1397,35 · 1394,84 · 1392,92 · 1375,89 · 903,22 · 230,11.

**Teste 4: resultado final.**

| Grupo | Centroide (x_km; y_km) | Entregas | Primeiros ids |
| --- | --- | --- | --- |
| 1 | (14,49; 6,15) | 20 | 1, 4, 5, 7, 14 |
| 2 | (3,66; 5,47) | 22 | 3, 8, 12, 13, 17 |
| 3 | (8,71; 14,90) | 18 | 2, 6, 9, 10, 11 |

J final = 230,11 e silhueta média s̄ = 0,756.

**Teste 5: cotovelo e silhueta.** Para cada k, use os k primeiros ids de `INIT_LOG`.

| k | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| J | 2508,07 | 1450,19 | 230,11 | 205,77 | 188,09 | 173,49 |
| s̄ | — | 0,480 | 0,756 | 0,609 | 0,611 | 0,426 |

> **Para discutir no relatório.** Rode de novo com k = 3, mas começando pelas entregas 3, 4 e 13. O algoritmo para depois de 6 atualizações com J = 1257,60, quase 5,5 vezes o valor anterior, e grupos de 10, 38 e 12 entregas. Explique por que isso acontece e como evitar.

## Dataset 2: vendas de produtos esportivos (enunciado)

Uma loja de artigos esportivos quer segmentar seus 120 clientes para criar campanhas diferentes. Configuração:

- Colunas de X: `idade`, `compras_ano`, `ticket_medio` (R<span>$</span>) e `pct_online` (% das compras feitas pelo site). A coluna `categoria_favorita` não entra em X; use-a só para interpretar os grupos.
- Padronize as quatro colunas (z-score).
- Lista de inicialização: `INIT_VENDAS = [10, 13, 11, 4, 18, 3, 5, 15]`.
- Salvo indicação em contrário, use k = 4 (centroides iniciais: clientes 10, 13, 11 e 4).

Para obter os valores usados nas perguntas, rode `relatorio("vendas")` no template: a função imprime μ e σ das colunas, os tamanhos dos grupos, o histórico de J, os centroides nas unidades originais, a silhueta e a curva de J(k) para k de 1 a 8, além da comparação com a solução sem padronização. Interprete os perfis encontrados e responda às perguntas no sistema da disciplina.

## Dataset 3: produtos bancários (enunciado)

Um banco quer agrupar 160 clientes para oferecer pacotes de produtos adequados a cada perfil. Configuração:

- Colunas de X: `idade`, `renda_mensal` (R<span>$</span>), `saldo_medio` (R<span>$</span>), `n_produtos` (produtos contratados) e `uso_limite_credito_pct` (% do limite do cartão usado).
- Padronize as cinco colunas (z-score).
- Lista de inicialização: `INIT_BANCO = [13, 18, 20, 21, 9, 12, 7, 24]`.
- Salvo indicação em contrário, use k = 4 (centroides iniciais: clientes 13, 18, 20 e 21).

Para obter os valores usados nas perguntas, rode `relatorio("banco")` no template: a função imprime μ e σ das colunas, os tamanhos dos grupos, o histórico de J, os centroides nas unidades originais, a silhueta e a curva de J(k) para k de 1 a 8, além da comparação com a solução sem padronização. Interprete os perfis encontrados e responda às perguntas no sistema da disciplina.

## Entrega e avaliação

Os códigos deverão estar em um repositório no GitHub, e o link do repositório deverá ser preenchido na planilha do curso.
