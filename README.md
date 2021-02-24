# RAPID-DTW

Códigos referentes ao artigo: "An Efficient Solution to Generate Meta-features for Classification with Remote Sensing Time Series" apresentado no Brazilian Symposium on Geoinformatics (GeoInfo), 2020. ISSN 2179-4847. Disponível em: http://urlib.net/rep/8JMKD3MGPDW34P/43PLBGP
    

# Algoritmos

Esta implementação utiliza como base o código do P-TWDTW, disponível em: https://github.com/savioteles/ptwdtw

As duas versões do algortimo precisam dos mesmos requisitos para a execução. O Rapid-DTW possui diversas configurações. Por padrão ele está configurado para trabalhar com séries temporais de tamanhos pequenos (24 observações) e janelas de tamanho 2.

## Parâmetros do código
O código principal está em src/csv.cpp e para executar o código precisa dos seguintes parâmetros:

- --help ou -h: "Tela de ajuda"
- --directory ou -d: "Caminho para o diretório com os arquivos usados como entrada de dados das séries temporais e dos padrões".
- --benchmark ou -b: "Caminho para o arquivo que define os parâmetros de execução do benchmark ou benchmarks dos testes".
- --repeat ou -r: "Número de repetições que serão executados os testes definidos no arquivo de benchmark".
- --cpu ou -c: "Indica que o benchmark irá ser executado usando os núcleos da CPU".
- --gpu ou -g: "Indica que o benchmark irá ser executado usando os núcleos da GPU".
- --test ou -t: "Flag que indica para gerar dados de testes com grandes séries temporais");

## Exemplo de execução no Nsight

Para rodar dentro do Nsight um primeiro exemplo basta mandar executar o projeto com os argumentos abaixo (em "Arguments") na GPU (-g) repetindo 10 vezes (-r 10) com os arquivos de entrada da pasta *files* e do benchmark em *benchmarks/fixed_num_ts_benchmark.csv*

```
-d "files/" -b "benchmarks/fixed_num_ts_benchmark.csv" -r 10 -g
```

**Importante:** configurar as flags -lboost_date_time e -lboost_program_options clicando com o botão direito no projeto e ir em: Properties->Build->Settings->Tool Settings->Libraries e adicionar "boost_date_time" e "boost_program_options". As bibliotecas precisam estar instaladas na máquina.


## TheadsWay

Um programa simples usado com auxilar para entender o comportamento das threads dentro da matriz.


## Qualquer dúvida na execução do programa entrar em contato com urzedabr@ufg.br
