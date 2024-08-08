# Orientações

Link documentacao - https://www.pywhy.org/dowhy/v0.9.1/example_notebooks/dowhy_simple_example.html#Replacing-treatment-with-a-random-(placebo)-variable

Link Exemplo Tacla - https://www.linkedin.com/pulseapplying-causal-inference-placebo-tests-infer-real-just-jancovic-s3bjf/

 - Para criar relações:
        
        model = CausalModel(
            data=df,
            treatment = "Variavel de tratamento",
            outcome = "Variavel de interesse",  
            common_causes = "[ Variaveis que afetam tanto o tratamento quanto o resultado ]",
            graph="""
                digraph {
                    Relacoes -> causais;
                    Outra -> relacao
                }
            """
        )

## METODOS DO MODELO 

    methods = [
        "backdoor.propensity_score_matching",
        "backdoor.propensity_score_weighting",
        "backdoor.propensity_score_stratification",
        "backdoor.linear_regression",
        "iv.two_stage_least_squares",
        "frontdoor.adjustment",
        "dml.dml",
        "regression_discontinuity",
        "difference_in_differences"
    ]


## Bibliotecas

    - pip install dowhy pandas graphviz matplotlib numpy


## Warnings por desatualização 
    
    (1) - (Linha 59) Precisa modificar o código da biblioteca dowhy localmente, especificamente no arquivo regression_estimator.py
        Linha original que gera o aviso:
            intercept_parameter = self.model.params[0]
            
        Linha modificada para evitar o aviso:
            intercept_parameter = self.model.params.iloc[0]
    
    (2) - (Linha 258) O aviso está sendo gerado na linha 258 do arquivo causal_estimator.py.
        Linha original que gera o aviso:
            by_effect_mods = self._data.groupby(effect_modifier_names)
    
        Linha modificada para evitar o aviso:
            by_effect_mods = self._data.groupby(effect_modifier_names, observed=False)