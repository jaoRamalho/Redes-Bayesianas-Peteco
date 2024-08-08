import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dowhy import CausalModel
import dowhy.datasets
import pydot

#  -------------------------- [ FILTRANDO DADOAS] --------------------------------------------

# Upando um banco de dados de um hotel
dataset = pd.read_csv('https://raw.githubusercontent.com/Sid-darthvader/DoWhy-The-Causal-Story-Behind-Hotel-Booking-Cancellations/master/hotel_bookings.csv')

# Total de estadias
dataset['total_stay'] = dataset['stays_in_week_nights'] + dataset['stays_in_weekend_nights']

# Numero total de hospedes
dataset['guests'] = dataset['adults'] + dataset['children'] + dataset['babies']

# Atribuindo 1 para reservas que trocaram de quarto e 0 para as que não trocaram
dataset['different_room_assigned'] = 0
slice_indices = dataset['reserved_room_type'] != dataset['assigned_room_type']
dataset.loc[slice_indices,'different_room_assigned'] = 1

# Deletando colunas que não serão utilizadas
dataset = dataset.drop(['stays_in_week_nights', 'stays_in_weekend_nights', 'adults', 'children', 'babies', 'reserved_room_type','assigned_room_type'],axis=1)

print("Soma de valores nulos : " + str(dataset.isnull().sum())) # Country,Agent,Company contain 488,16340,112593 missing entries

# Deletando colunas com muitos valores nulos
dataset = dataset.drop(['agent','company'],axis=1)

# Removendo linhas com valores nulos e substituindo valores nulos por valores mais frequentes
dataset['country']= dataset['country'].fillna(dataset['country'].mode()[0])

dataset = dataset.drop(['reservation_status','reservation_status_date','arrival_date_day_of_month'],axis=1)
dataset = dataset.drop(['arrival_date_year'],axis=1)
dataset = dataset.drop(['distribution_channel'], axis=1)

# Substituindo 1 por True e 0 por False
dataset['different_room_assigned']= dataset['different_room_assigned'].replace(1,True)
dataset['different_room_assigned']= dataset['different_room_assigned'].replace(0,False)
dataset['is_canceled']= dataset['is_canceled'].replace(1,True)
dataset['is_canceled']= dataset['is_canceled'].replace(0,False)
dataset.dropna(inplace=True)

# Filtra o dataset para manter apenas as linhas onde a coluna deposit_type é igual a "No Deposit"
dataset = dataset[dataset.deposit_type=="No Deposit"]

# Agrupa o dataset por deposit_type e is_canceled e conta o número de entradas
print ("Não depositado com antecendia e cancelaram : " + str(dataset.groupby(['deposit_type','is_canceled']).count()))

# ---------------------------------------------------------------------------------------------------------------

dataset_copy = dataset.copy(deep=True)


print("\n -------------------------------------- [ COLUNAS ] --------------------------------------")
print(dataset.columns)
print("-----------------------------------------------------------------------------------------")


print("\n ------------------------------------ [ LINHAS 5 A 19 DO BANCO DE DADOS ] ------------------------------------ ")
print(dataset.iloc[:, 5:20].head(10));
print(" -------------------------------------------------------------------------------------------")

"""

# --------------------------------------------------------------------------------------------------------------------

soma = 0
for i in range(1,10000):
    counts_i = 0
    rdf = dataset.sample(1000)
    counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
    soma+= counts_i

print("\n Media de cancelamentos onde houve troca de quartos : " + str(soma/10000))



# Expected Count when there are no booking changes
soma = 0
for i in range(1,10000):
    counts_i = 0
    rdf = dataset[dataset["booking_changes"]==0].sample(1000)
    counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
    soma += counts_i
print("Media de cancelamentos onde não houve mudança de reserva : " + str(soma/10000))


# Expected Count when there are booking changes = 66.4%
soma = 0
for i in range(1,10000):
    counts_i = 0
    rdf = dataset[dataset["booking_changes"]>0].sample(1000)
    counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
    soma += counts_i
print("Media de cancelamento onde houve mudança de reserva : " + str(soma/10000))
"""

# ------------------------------------------- [ TRATANDO DADOS ] ------------------------------------------------------


#   Consideramos quais fatores causam o cancelamento de uma reserva de hotel. 
#   Esta análise é baseada em um conjunto de dados de reservas de hotel de Antonio, Almeida e Nunes (2019) . 
#   No GitHub, o conjunto de dados está disponível em rfordatascience/tidytuesday .



# ETAPA 1: CRIE O GRAFICO CAUSAL

causal_graph = """
digraph {
    different_room_assigned [label="Different Room Assigned"];
    is_canceled [label="Booking Cancelled"];
    booking_changes [label="Booking Changes"];
    previous_bookings_not_canceled [label="Previous Booking Retentions"];
    days_in_waiting_list [label="Days in Waitlist"];
    lead_time [label="Lead Time"];
    market_segment [label="Market Segment"];
    country [label="Country"];
    U [label="Unobserved Confounders", observed="no"];
    is_repeated_guest;
    total_stay;
    guests;
    meal;
    hotel;
    U -> different_room_assigned;
    U -> required_car_parking_spaces;
    U -> guests;
    U -> total_stay;
    U -> total_of_special_requests;
    market_segment -> lead_time;
    lead_time -> is_canceled;
    country -> lead_time;
    different_room_assigned -> is_canceled;
    country -> meal;
    lead_time -> days_in_waiting_list;
    days_in_waiting_list -> is_canceled;
    days_in_waiting_list -> different_room_assigned;
    previous_bookings_not_canceled -> is_canceled;
    previous_bookings_not_canceled -> is_repeated_guest;
    is_repeated_guest -> different_room_assigned;
    is_repeated_guest -> is_canceled;
    total_stay -> is_canceled;
    guests -> is_canceled;
    booking_changes -> different_room_assigned;
    booking_changes -> is_canceled;
    hotel -> different_room_assigned;
    hotel -> is_canceled;
    required_car_parking_spaces -> is_canceled;
    total_of_special_requests -> booking_changes;
    total_of_special_requests -> is_canceled;
    country -> hotel;
    country -> required_car_parking_spaces;
    country -> total_of_special_requests;
    market_segment -> hotel;
    market_segment -> required_car_parking_spaces;
    market_segment -> total_of_special_requests;
}
"""

# ETAPA 2: CRIE O MODELO CAUSAL
model= CausalModel(
    data = dataset,
    treatment="different_room_assigned",
    outcome='is_canceled',
    graph=causal_graph
)

model.view_model()


# ETAPA 3: IDENTIFICAR O EFEITO CAUSAL
print ("/n ------------------------------------ [ IDENTIFICAR O EFEITO CAUSAL ] ------------------------------------ /n")
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)
print("-----------------------------------------------------------------------------------------")


# ETAPA 4: ESTIMAR O EFEITO CAUSAL
print ("/n ------------------------------------ [ ESTIMAR O EFEITO CAUSAL ] ------------------------------------ /n")
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_weighting",target_units="ate")
# ATE = Average Treatment Effect
# ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
# ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)
print(estimate)
print("-----------------------------------------------------------------------------------------")


# ETAPA 5: REFUTAR DADOS
# TENHO QUE TESTAR AINDA..
