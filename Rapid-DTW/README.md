# Algoritmo PTWDTW

Versão paralela do algoritmo TWDTW (Time-Weighted Dynamic Time Warping) na GPU e CPU.

## Parâmetros do código
O código principal está em src/csv.cpp e para executar o código precisa dos seguintes parâmetros:

- --help ou -h: "Tela de ajuda"
- --directory ou -d: "Caminho para o diretório com os arquivos usados como entrada de dados das séries temporais e dos padrões".
- --benchmark ou -b: "Caminho para o arquivo que define os parâmetros de execução do benchmark ou benchmarks dos testes".
- --repeat ou -r: "Número de repetições que serão executados os testes definidos no arquivo de benchmark. Este parâmetro é importante para tomada de tempo para descartar possíveis casos discrepantes.".
- --cpu ou -c: "Indica que o benchmark irá ser executado usando os núcleos da CPU".
- --gpu ou -g: "Indica que o benchmark irá ser executado usando os núcleos da GPU".
- --test ou -t: "Flag que indica para gerar dados de testes com grandes séries temporais");

## Exemplo de execução no Nsight

Para rodar dentro do Nsight um primeiro exemplo basta mandar executar o projeto com os argumentos abaixo (em "Arguments") na CPU (-c) repetindo 10 vezes (-r 10) com os arquivos de entrada da pasta *files* e do benchmark em *benchmarks/fixed_num_ts_benchmark.csv*

```
-d "files/" -b "benchmarks/fixed_num_ts_benchmark.csv" -r 10 -c
```

**Importante:** configurar as flags -lboost_date_time e -lboost_program_options clicando com o botão direito no projeto e ir em: Properties->Build->Settings->Tool Settings->Libraries e adicionar "boost_date_time" e "boost_program_options". As bibliotecas precisam estar instaladas na máquina.

