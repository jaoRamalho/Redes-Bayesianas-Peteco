import pandas as pd
import numpy as np

from dowhy import CausalModel
import dowhy.datasets

data = dowhy.datasets.linear_dataset(
    beta=10,
    num_common_causes=5,
    num_instruments = 2,
    num_effect_modifiers=1,
    num_samples=100,
    treatment_is_binary=True,
    stddev_treatment_noise=10,
    num_discrete_common_causes=1
)

df = data["df"]


# Estimar o efeito causal usando diferentes métodos
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


# Definir o modelo causal
model = CausalModel(
    data = df,
    treatment=data["treatment_name"],
    outcome=data["outcome_name"],
    graph=data["gml_graph"]
)

print("\n ------------------------------------ Teste ------------------------------------  \n")


# Visualizar o gráfico causal
model.view_model()


# Identificar o efeito causal
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
#print(identified_estimand)

estimate = model.estimate_effect( identified_estimand,  method_name = methods[3] ) 


print("\n ------------------------------------ Bando de dados ------------------------------------  \n")
print(df.head(10));
print("\n ----------------------------------------------------------------------------------------  \n")




print("\n ------------------------------------ Estimativa ------------------------------------  \n")
print(estimate)
print("\n ----------------------------------------------------------------------------------------  \n")

"""
print("\n ------------------------------------- Refutando modelo ----------------------------------  \n")
print("Falha = p < 0.05 \n")
print("\n1 - Invariant transformations - Mudanças nos dados que não deveriam alterar a estimativa. Qualquer estimador cujo resultado varie significativamente entre os dados originais e os dados modificados falha no teste: \n")

print("\n - Random Common Cause: Adding a random common cause variable: \n")
res_random=model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause", show_progress_bar=True)
print(res_random)

print("\n - Data Subset Refuter: Removing a subset of the data: \n")
res_subset=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter", show_progress_bar=True, subset_fraction=0.9)
print(res_subset)


print("\n2- Nullifying transformations: after the data change, the causal true estimate is zero. Any estimator whose result varies significantly from zero on the new data fails the test: \n")
print("\n - Placebo Treatment: Replacing treatment with a random (placebo) variable \n")
res_placebo=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter", show_progress_bar=True, placebo_type="permute")
print(res_placebo)

"""

res_unobserved=model.refute_estimate(identified_estimand, estimate, method_name="add_unobserved_common_cause",
                                     confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linear",
                                    effect_strength_on_treatment=0.01, effect_strength_on_outcome=0.02)
print(res_unobserved)

res_unobserved_range=model.refute_estimate(identified_estimand, estimate, method_name="add_unobserved_common_cause",
                                     confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linear",
                                    effect_strength_on_treatment=np.array([0.001, 0.005, 0.01, 0.02]), effect_strength_on_outcome=0.01)
print(res_unobserved_range)

res_unobserved_range=model.refute_estimate(identified_estimand, estimate, method_name="add_unobserved_common_cause",
                                           confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linear",
                                           effect_strength_on_treatment=[0.001, 0.005, 0.01, 0.02],
                                           effect_strength_on_outcome=[0.001, 0.005, 0.01,0.02])
print(res_unobserved_range)

res_unobserved_auto = model.refute_estimate(identified_estimand, estimate, method_name="add_unobserved_common_cause",
                                           confounders_effect_on_treatment="binary_flip", confounders_effect_on_outcome="linear")
print(res_unobserved_auto)
