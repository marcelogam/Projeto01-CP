# Projeto01-CP
Repositório para o projeto 01 da matéria de computação paralela
link: https://github.com/marcoscastro/kmeans/blob/master/kmeans.cpp

# Visão Geral do Algoritmo K-Means

O algoritmo K-Means é um método de aprendizado não supervisionado usado para agrupar pontos de dados em K clusters com base em suas características. O objetivo é minimizar a variância dentro dos clusters e maximizar a variância entre os clusters.

O algoritmo funciona iterativamente através dos seguintes passos:

Inicialização: Seleciona K centroides iniciais aleatoriamente.
Atribuição de Clusters: Atribui cada ponto ao cluster cujo centroide está mais próximo.
Atualização de Centroides: Recalcula os centroides como a média dos pontos atribuídos a cada cluster.
Convergência: Repete os passos 2 e 3 até que as atribuições não mudem ou seja alcançado o número máximo de iterações.


# Explicação Detalhada do Código:
O código está estruturado em três classes principais: Point, Cluster e KMeans. Além disso, há a função main que coordena a execução do programa.

Classe Point:
Objetivo: Representa um ponto de dados no espaço n-dimensional.
    Principais Atributos:
        id_point: Identificador único do ponto.
        id_cluster: Identificador do cluster ao qual o ponto pertence.
        values: Vetor que armazena os valores das características do ponto.
        name: Nome opcional do ponto (pode ser usado para identificação).
    Principais Métodos:
        Construtor: Inicializa um ponto com ID, valores e nome opcional.
        setCluster(int id_cluster): Define o cluster ao qual o ponto pertence.
        getCluster(): Retorna o ID do cluster ao qual o ponto pertence.
        getValue(int index): Retorna o valor da característica no índice especificado.

Classe Cluster:
Objetivo: Representa um cluster, que é um grupo de pontos e seu centroide.
    Principais Atributos:
        id_cluster: Identificador do cluster.
        central_values: Vetor que armazena os valores do centroide do cluster.
        points: Vetor que armazena os pontos pertencentes ao cluster.
    Principais Métodos:
        Construtor: Inicializa um cluster com um ID e um ponto inicial (que define o centroide inicial).
        addPoint(Point point): Adiciona um ponto ao cluster.
        removePoint(int id_point): Remove um ponto do cluster com base no ID.
        getCentralValue(int index): Retorna o valor do centroide no índice especificado.
        setCentralValue(int index, double value): Define o valor do centroide no índice especificado.
        getPoint(int index): Retorna o ponto no índice especificado.
        getTotalPoints(): Retorna o número total de pontos no cluster.

Classe KMeans:
Objetivo: Implementa o algoritmo K-Means.
    Principais Atributos:
        K: Número de clusters desejados.
        total_values: Número de características em cada ponto.
        total_points: Número total de pontos de dados.
        max_iterations: Número máximo de iterações permitidas.
        clusters: Vetor que armazena os clusters.
    Principais Métodos:
        Construtor: Inicializa os parâmetros do algoritmo.
        getIDNearestCenter(Point point): Calcula o centroide mais próximo de um ponto dado (usando distância Euclidiana) e retorna o ID do cluster correspondente.
        run(vector<Point> & points): Método principal que executa o algoritmo K-Means nos pontos fornecidos.


## OpenMP
# Comparação entre o Código Original e o Código Paralelizado

1. Modificação da Função main para Aceitar o Número de Threads

    - Leitura do Número de Threads:
        Foi adicionado um bloco de código para ler o número de threads a partir dos argumentos de linha de comando (argv).
        Para permitir que o usuário especifique o número de threads (1, 2, 4 ou 8) ao executar o programa, facilitando a comparação de desempenho entre diferentes níveis de paralelismo.
    - Definição do Número de Threads:
        Uso de omp_set_num_threads(num_threads);
        Essa função configura o número de threads que o OpenMP utilizará nas regiões paralelas.
    - Padrão:
        Se o usuário não fornecer o número de threads, o programa usa 1 thread por padrão.
    - Impacto: 
        Permite controlar dinamicamente o número de threads utilizados pelo programa, essencial para testar o desempenho em diferentes configurações.

2. Paralelização do Loop de Associação de Pontos aos Clusters

    - Adição da Diretiva #pragma omp parallel for:
        Foi adicionada a diretiva para paralelizar o loop que associa cada ponto ao centro mais próximo.
        Cada iteração do loop é independente, permitindo que o processamento seja distribuído entre múltiplas threads.
    - Uso da Variável new_clusters:
        Em vez de atualizar os clusters dentro do loop paralelo (o que poderia causar condições de corrida), armazenamos temporariamente o novo cluster de cada ponto no vetor new_clusters.
    - Operação Atômica para done:
        Utilizamos #pragma omp atomic write para atualizar a variável done de forma thread-safe.
        Para evitar condições de corrida ao atualizar uma variável compartilhada entre threads.
    - Reatribuição dos Pontos aos Clusters:
        Após o loop paralelo, em uma seção sequencial, limpamos os pontos dos clusters antigos e reatribuímos os pontos com base em new_clusters.
        Garante a consistência dos dados e evita problemas de concorrência.

3. Paralelização da Recomputação dos Centroides dos Clusters

    - Adição da Diretiva #pragma omp parallel for no Loop Externo:
        Paralelizamos o loop que itera sobre cada cluster (i).
        Para distribuir o processamento da recomputação dos centroides entre múltiplas threads.
    - Paralelização da Soma Interna:
        Adicionamos #pragma omp parallel for reduction(+ : sum) no loop que soma os valores dos pontos dentro de cada cluster para cada dimensão (j).
        Para paralelizar a soma dos valores dos pontos em cada dimensão, utilizando redução para evitar condições de corrida na variável sum.
    - Uso de Redução (reduction):
        A redução garante que múltiplas threads possam acumular valores em uma variável compartilhada de forma segura.
    - Impacto
        A recomputação dos centroides é acelerada pela execução em paralelo, melhorando o desempenho geral do algoritmo.

4.  Adição de Funções Auxiliares para Suporte ao Paralelismo

    - Adição da Função clearPoints():
        Foi adicionada uma função para limpar os pontos atribuídos ao cluster.
        Necessário para remover os pontos dos clusters antes de reatribuí-los, garantindo que a estrutura de dados esteja consistente após cada iteração.
    - Impacto:
        Facilita a reatribuição dos pontos aos clusters sem a necessidade de gerenciar manualmente a remoção dos pontos, evitando possíveis erros e simplificando o código.

5. Comentários e Mensagens Informativas

    - Comentário sobre a Impressão dos Resultados:
        A seção que imprime os detalhes dos clusters foi comentada.
        A impressão no console pode ser uma operação custosa em termos de tempo, especialmente com grandes conjuntos de dados. Comentando essa parte, o tempo de execução é reduzido.

6. Medição do Tempo de Execução, Controle de Aleatoriedade

    - Inclusão do número de threads na saída de tempo de execução.
    - Controle de Aleatoriedade
        A aleatoriedade na escolha dos centroides iniciais pode levar a resultados diferentes entre execuções, o que deve ser considerado ao analisar os resultados. -> srand(time(NULL));
        Fixamos a semente aleatória (srand(0);) para garantir reprodutibilidade nos testes.




## MPI











Considerações para a Apresentação
Explique o Contexto das Modificações:

Foque em como o paralelismo foi introduzido para melhorar o desempenho.
Discuta os desafios enfrentados, como condições de corrida e necessidade de sincronização.
Destaque os Benefícios do Paralelismo:

Apresente comparações de tempos de execução entre a versão sequencial e as versões paralelas com diferentes números de threads.
Explique como o OpenMP simplificou a paralelização do código.
Mencione as Boas Práticas Adotadas:

Uso de reduções para variáveis compartilhadas.
Evitar atualizações concorrentes em estruturas de dados complexas dentro de regiões paralelas.
Aborde Possíveis Limitações:

Discuta a escalabilidade do algoritmo e onde podem surgir gargalos mesmo com paralelismo.
Mencione a influência da escolha dos centroides iniciais e como a aleatoriedade pode afetar os resultados.
Conclua com Lições Aprendidas:

A importância de entender a lógica do algoritmo para paralelizá-lo efetivamente.
Como o paralelismo pode ser aplicado para melhorar o desempenho em problemas de processamento intensivo.

