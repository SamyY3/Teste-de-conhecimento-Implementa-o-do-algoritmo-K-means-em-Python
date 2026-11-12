"""
Trabalho: Implementação manual do K-means
Matemática Computacional - Prof. Dr. Alexsandro O. A.

Nome(s): Maria Taliciane Pereira da Silva
Regras:
- Use apenas NumPy, pandas e Matplotlib.
- É proibido usar sklearn, scipy.cluster ou qualquer K-means / silhueta prontos.
- Siga exatamente o pseudocódigo e as convenções do enunciado.

Como usar:
- Implemente as funções marcadas com "TODO" (remova o `raise NotImplementedError`).
- Rode `python kmeans_template.py`. Os testes do dataset 1 mostram OK ou FALHOU.
- Depois, use as saídas dos datasets 2 e 3 para responder aos quizzes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuração dos datasets (NÃO ALTERE)
# ---------------------------------------------------------------------------

INIT_LOG = [2, 4, 6, 8, 10, 12]
INIT_VENDAS = [10, 13, 11, 4, 18, 3, 5, 15]
INIT_BANCO = [13, 18, 20, 21, 9, 12, 7, 24]

DATASETS = {
    "logistica": {
        "arquivo": "entregas_correios.csv",
        "id": "id_entrega",
        "colunas": ["x_km", "y_km"],
        "padronizar": False,
        "init": INIT_LOG,
        "k": 3,
    },
    "vendas": {
        "arquivo": "vendas_esportivas.csv",
        "id": "id_cliente",
        "colunas": ["idade", "compras_ano", "ticket_medio", "pct_online"],
        "padronizar": True,
        "init": INIT_VENDAS,
        "k": 4,
    },
    "banco": {
        "arquivo": "clientes_banco.csv",
        "id": "id_cliente",
        "colunas": ["idade", "renda_mensal", "saldo_medio", "n_produtos",
                    "uso_limite_credito_pct"],
        "padronizar": True,
        "init": INIT_BANCO,
        "k": 4,
    },
}


# ---------------------------------------------------------------------------
# Funções fornecidas (prontas)
# ---------------------------------------------------------------------------

def carregar_dados(arquivo, colunas, coluna_id):
    """Lê o CSV e devolve (X, df, ids).

    X   : matriz n x d (float) com as colunas escolhidas
    df  : DataFrame completo (para interpretar os grupos)
    ids : vetor n com os ids de cada linha
    """
    df = pd.read_csv(arquivo)
    X = df[colunas].to_numpy(dtype=float)
    ids = df[coluna_id].to_numpy()
    return X, df, ids


def contar_mudancas(r_a, r_b):
    """Conta quantos dados mudam de grupo entre duas soluções.

    Cada grupo de r_b é associado ao grupo de r_a com o qual mais compartilha
    dados (a numeração dos grupos pode ser diferente nas duas soluções).
    """
    mudancas = 0
    for g in np.unique(r_b):
        membros = r_a[r_b == g]
        grupo_majoritario = np.bincount(membros).argmax()
        mudancas += int((membros != grupo_majoritario).sum())
    return mudancas


# ---------------------------------------------------------------------------
# Funções a implementar
# ---------------------------------------------------------------------------

def padronizar(X):
    """TODO: padronização z-score, coluna por coluna.

    Use a média e o desvio-padrão POPULACIONAL (divisão por n, ddof=0).
    Retorne (Z, mu, sigma), em que mu e sigma são vetores de tamanho d.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    sigma_seguro = np.where(sigma == 0, 1.0, sigma)
    Z = (X - mu) / sigma_seguro
    return Z, mu, sigma_seguro


def centroides_iniciais(X, ids, lista_ids):
    """TODO: monte a matriz C (k x d) com as linhas de X cujos ids estão em
    lista_ids, NA ORDEM da lista (o 1º id vira o grupo 1, e assim por diante).
    """
    X = np.asarray(X, dtype=float)
    ids = np.asarray(ids)
    lista_ids = list(lista_ids)
    posicoes = {valor: i for i, valor in enumerate(ids.tolist())}
    ausentes = [valor for valor in lista_ids if valor not in posicoes]
    if ausentes:
        raise ValueError(f"IDs de inicialização não encontrados: {ausentes}")
    return X[[posicoes[valor] for valor in lista_ids]].copy()


def matriz_distancias(X, C):
    """TODO: devolva a matriz D (n x k) com D[i, k] = ||x_i - c_k||^2.

    Dica: use broadcasting, X[:, None, :] - C[None, :, :], sem laços sobre i.
    """
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    diferencas = X[:, None, :] - C[None, :, :]
    return np.sum(diferencas ** 2, axis=2)


def atribuir(D):
    """TODO: devolva o vetor r (n) com o índice da menor entrada de cada linha de D.

    Em caso de empate, vence o menor índice (np.argmin já faz isso).
    """
    D = np.asarray(D, dtype=float)
    return np.argmin(D, axis=1)


def atualizar_centroides(X, r, C):
    """TODO: devolva a nova matriz de centroides (k x d).

    Cada c_k é a média dos x_i com r[i] == k.
    Se um grupo ficar vazio, seu centroide NÃO muda (use o valor de C).
    """
    X = np.asarray(X, dtype=float)
    r = np.asarray(r, dtype=int)
    C_novo = np.asarray(C, dtype=float).copy()
    for g in range(C_novo.shape[0]):
        membros = X[r == g]
        if len(membros) > 0:
            C_novo[g] = membros.mean(axis=0)
    return C_novo


def calcular_J(X, C, r):
    """TODO: devolva J = soma de ||x_i - c_{r_i}||^2 (um número float)."""
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    r = np.asarray(r, dtype=int)
    diferencas = X - C[r]
    return float(np.sum(diferencas ** 2))


def kmeans(X, C_inicial, max_iter=100):
    """TODO: implemente o pseudocódigo do enunciado.

    Retorne (r, C, historico_J, n_atualizacoes):
    - r              : vetor n com o grupo final de cada dado (0, 1, ..., k-1)
    - C              : matriz k x d com os centroides finais
    - historico_J    : lista com J calculado após cada atualização
    - n_atualizacoes : número de atualizações dos centroides
    """
    X = np.asarray(X, dtype=float)
    C = np.asarray(C_inicial, dtype=float).copy()
    r_anterior = None
    historico_J = []
    n_atualizacoes = 0

    for _ in range(max_iter):
        r = atribuir(matriz_distancias(X, C))
        if r_anterior is not None and np.array_equal(r, r_anterior):
            break
        C = atualizar_centroides(X, r, C)
        n_atualizacoes += 1
        historico_J.append(calcular_J(X, C, r))
        r_anterior = r.copy()

    return r, C, historico_J, n_atualizacoes


def silhueta(X, r):
    """TODO: devolva (s, s_media).

    s[i] = (b(i) - a(i)) / max(a(i), b(i)), com distâncias euclidianas SEM o
    quadrado. Se o grupo de i tiver um único elemento, use s[i] = 0.
    """
    X = np.asarray(X, dtype=float)
    r = np.asarray(r, dtype=int)
    n = X.shape[0]
    s = np.zeros(n, dtype=float)
    grupos = np.unique(r)
    if len(grupos) <= 1:
        return s, 0.0

    # Distâncias euclidianas entre todos os pares, sem elevar ao quadrado.
    diferencas = X[:, None, :] - X[None, :, :]
    distancias = np.sqrt(np.sum(diferencas ** 2, axis=2))

    indices_por_grupo = {g: np.flatnonzero(r == g) for g in grupos}
    for i in range(n):
        grupo_i = r[i]
        colegas = indices_por_grupo[grupo_i]
        if len(colegas) <= 1:
            s[i] = 0.0
            continue
        outros = colegas[colegas != i]
        a = float(distancias[i, outros].mean())
        b = min(float(distancias[i, indices].mean())
                for g, indices in indices_por_grupo.items() if g != grupo_i)
        denominador = max(a, b)
        s[i] = (b - a) / denominador if denominador > 0 else 0.0

    return s, float(s.mean())


# ---------------------------------------------------------------------------
# Rotinas de apoio (prontas, usam as suas funções)
# ---------------------------------------------------------------------------

def preparar(nome):
    """Carrega o dataset `nome` e devolve (X_usado, X_original, df, ids, mu, sigma)."""
    cfg = DATASETS[nome]
    X, df, ids = carregar_dados(cfg["arquivo"], cfg["colunas"], cfg["id"])
    if cfg["padronizar"]:
        Z, mu, sigma = padronizar(X)
        return Z, X, df, ids, mu, sigma
    return X, X, df, ids, np.zeros(X.shape[1]), np.ones(X.shape[1])


def rodar(nome, k=None, padronizado=True):
    """Roda o K-means no dataset `nome` com os k primeiros ids da lista de init."""
    cfg = DATASETS[nome]
    k = cfg["k"] if k is None else k
    Xu, Xo, df, ids, mu, sigma = preparar(nome)
    if not padronizado:
        Xu = Xo
    C0 = centroides_iniciais(Xu, ids, cfg["init"][:k])
    r, C, hist, n_upd = kmeans(Xu, C0)
    return {"X": Xu, "X_orig": Xo, "df": df, "ids": ids, "mu": mu, "sigma": sigma,
            "C0": C0, "r": r, "C": C, "historico_J": hist, "n_atualizacoes": n_upd}


def relatorio(nome, k_max=8, graficos=False):
    """Imprime as informações usadas nos quizzes dos datasets 2 e 3."""
    cfg = DATASETS[nome]
    print(f"\n{'=' * 70}\nDATASET: {nome}  (k = {cfg['k']}, init = {cfg['init'][:cfg['k']]})\n{'=' * 70}")
    res = rodar(nome)
    np.set_printoptions(suppress=True, precision=4)
    print("Colunas:", cfg["colunas"])
    print("mu    =", res["mu"])
    print("sigma =", res["sigma"])

    r0 = atribuir(matriz_distancias(res["X"], res["C0"]))
    print("Tamanhos após a 1ª atribuição:", np.bincount(r0, minlength=cfg["k"]))
    print("Atualizações até convergir:", res["n_atualizacoes"])
    print("Histórico de J:", np.round(res["historico_J"], 2))
    print("J final:", round(res["historico_J"][-1], 2))
    print("Tamanhos finais:", np.bincount(res["r"], minlength=cfg["k"]))

    C_orig = res["C"] * res["sigma"] + res["mu"]
    tabela = pd.DataFrame(C_orig, columns=cfg["colunas"])
    tabela.index = [f"grupo {g + 1}" for g in range(cfg["k"])]
    print("\nCentroides nas unidades originais:\n", tabela.round(2))

    df = res["df"].copy()
    df["grupo"] = res["r"] + 1
    for col in df.columns:
        if df[col].dtype == object:
            print(f"\nValor mais frequente de '{col}' por grupo:")
            print(df.groupby("grupo")[col].agg(lambda s: s.value_counts().index[0]))

    _, s_med = silhueta(res["X"], res["r"])
    print(f"\nSilhueta média (k = {cfg['k']}): {s_med:.3f}")

    print("\nk | J | silhueta média")
    Js, Ss = [], []
    for k in range(1, k_max + 1):
        rk = rodar(nome, k)
        Jk = calcular_J(rk["X"], rk["C"], rk["r"])
        Sk = silhueta(rk["X"], rk["r"])[1] if k > 1 else float("nan")
        Js.append(Jk)
        Ss.append(Sk)
        print(f"{k} | {Jk:.2f} | {Sk:.3f}")

    sem = rodar(nome, padronizado=False)
    print("\nSem padronização: tamanhos =", np.bincount(sem["r"], minlength=cfg["k"]),
          "| clientes que mudam de grupo =", contar_mudancas(res["r"], sem["r"]))

    if graficos:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(range(1, k_max + 1), Js, "o-")
        ax[0].set(title=f"Cotovelo - {nome}", xlabel="k", ylabel="J")
        ax[1].plot(range(2, k_max + 1), Ss[1:], "o-")
        ax[1].set(title=f"Silhueta média - {nome}", xlabel="k", ylabel="s médio")
        plt.tight_layout()
        plt.show()
    return res


# ---------------------------------------------------------------------------
# Testes do dataset 1 (entregas dos Correios)
# ---------------------------------------------------------------------------

def _confere(nome, obtido, esperado, tol=0.01):
    ok = np.allclose(np.asarray(obtido, dtype=float), np.asarray(esperado, dtype=float), atol=tol)
    print(f"[{'OK' if ok else 'FALHOU'}] {nome}")
    if not ok:
        print("    esperado:", np.round(esperado, 2), "\n    obtido:  ", np.round(obtido, 2))
    return ok


def testes_logistica(graficos=False):
    print("=" * 70 + "\nTESTES - DATASET 1 (entregas dos Correios)\n" + "=" * 70)
    X, _, df, ids, _, _ = preparar("logistica")

    C0 = centroides_iniciais(X, ids, [2, 4, 6])
    _confere("Teste 1a: centroides iniciais", C0, [[10.33, 14.50], [13.11, 6.39], [7.46, 16.07]])
    D = matriz_distancias(X, C0)
    _confere("Teste 1b: 5 primeiras linhas de D", D[:5],
             [[118.41, 9.44, 190.55], [0.00, 73.50, 10.70], [137.82, 102.41, 135.66],
              [73.50, 0.00, 125.62], [71.45, 0.90, 118.62]])
    r1 = atribuir(D)
    _confere("Teste 1c: grupos das 5 primeiras entregas (1 a 3)", r1[:5] + 1, [2, 1, 2, 2, 2])

    _confere("Teste 2a: tamanhos após a 1ª atribuição", np.bincount(r1, minlength=3), [8, 41, 11])
    C1 = atualizar_centroides(X, r1, C0)
    _confere("Teste 2b: centroides após a 1ª atualização", C1,
             [[9.71, 14.19], [8.99, 5.77], [7.35, 14.67]])
    _confere("Teste 2c: J após a 1ª atualização", calcular_J(X, C1, r1), 1494.25)

    r, C, hist, n_upd = kmeans(X, C0)
    _confere("Teste 3a: número de atualizações", n_upd, 8)
    _confere("Teste 3b: histórico de J", hist,
             [1494.25, 1406.25, 1397.35, 1394.84, 1392.92, 1375.89, 903.22, 230.11])

    _confere("Teste 4a: centroides finais", C, [[14.49, 6.15], [3.66, 5.47], [8.71, 14.90]])
    _confere("Teste 4b: tamanhos finais", np.bincount(r, minlength=3), [20, 22, 18])
    _confere("Teste 4c: silhueta média", silhueta(X, r)[1], 0.756, tol=0.001)

    Js, Ss = [], []
    for k in range(1, 7):
        rk = rodar("logistica", k)
        Js.append(calcular_J(rk["X"], rk["C"], rk["r"]))
        Ss.append(silhueta(rk["X"], rk["r"])[1] if k > 1 else 0.0)
    _confere("Teste 5a: J para k = 1..6", Js, [2508.07, 1450.19, 230.11, 205.77, 188.09, 173.49])
    _confere("Teste 5b: silhueta para k = 2..6", Ss[1:], [0.480, 0.756, 0.609, 0.611, 0.426], tol=0.001)

    if graficos:
        plt.figure(figsize=(5, 5))
        for g in range(3):
            plt.scatter(X[r == g, 0], X[r == g, 1], label=f"grupo {g + 1}")
        plt.scatter(C[:, 0], C[:, 1], marker="x", s=150, c="k", label="centroides")
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.title("Entregas por região")
        plt.legend()
        plt.show()


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        testes_logistica(graficos=False)
        relatorio("vendas", graficos=False)
        relatorio("banco", graficos=False)
    except NotImplementedError as e:
        print(f"\nFunção ainda não implementada: {e}()")
