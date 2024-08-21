# Driven Data - Richter's Predictor: Modeling Earthquake Damage

## Background
In this notebook, I explore a dataset from the [DrivenData competition](https://www.drivendata.org/competitions/57/nepal-earthquake/page/134/) focused on predicting the damage levels sustained by buildings during the 2015 Gorkha earthquake in Nepal. This catastrophic event resulted in widespread destruction, and the extensively collected data offers a comprehensive look at the impact. The dataset, one of the largest post-disaster collections ever assembled, includes critical information on the buildings' structure, household conditions, and socio-economic-demographic statistics. The objective is to predict the damage_grade—an ordinal variable indicating the severity of the damage, categorized into three grades: low damage (1), medium damage (2), and complete destruction (3). The dataset comprises 39 columns, with building_id serving as a unique identifier and the remaining 38 features detailing various aspects of the buildings.

This submission placed in the 32nd position on the leaderboard (DSIF13), ranking within the top 1.5% of [submissions](https://www.drivendata.org/competitions/57/nepal-earthquake/leaderboard/).

## Approach
__Ensembling GBMs__:

In this project, Gradient Boosting Machines (GBMs) are employed to predict building damage, utilizing classifier variants of these GBMs. Specifically, XGBoost, CatBoost, and LightGBM are trained on the dataset. Small perturbations are introduced during model construction to generate multiple versions of these models. An ensemble approach is then applied, where the mode of the predicted damage grades is used as the final prediction.

__Auto-encoder to extract latent geographical relationships__:

The dataset includes geographical features encoded hierarchically as integers. The top geographical feature has 30 possible values, while the middle and bottom features have 1,427 and 12,567 possible values, respectively. Aside from the hierarchy, there is no indication of proximity or relationships between the different geographical encodings within each layer. For an earthquake-related problem, buildings in close proximity are likely to experience similar levels of damage. In this project, an autoencoder is used to construct a representation of these geographical proximities, training on other features (e.g. building characteristics) in the dataset as well as the target variables. Hyperparameter tuning is performed carefully to avoid inadvertently using labels from the validation/test set during training. The resulting latent variables are extracted for model training.

## Dataset
The dataset is freely available at [DrivenData](#https://www.drivendata.org/competitions/57/nepal-earthquake/). The data description is reproduced below for reference.

| Feature                                      | Type        | Description                                                                                           |
|----------------------------------------------|-------------|-------------------------------------------------------------------------------------------------------|
| geo_level_1_id                              | int         | Geographic region in which the building exists, level 1 (0-30)                                       |
| geo_level_2_id                              | int         | Geographic region in which the building exists, level 2 (0-1427)                                     |
| geo_level_3_id                              | int         | Geographic region in which the building exists, level 3 (0-12567)                                    |
| count_floors_pre_eq                         | int         | Number of floors in the building before the earthquake                                               |
| age                                         | int         | Age of the building in years                                                                         |
| area_percentage                             | int         | Normalized area of the building footprint                                                             |
| height_percentage                           | int         | Normalized height of the building footprint                                                           |
| land_surface_condition                      | categorical | Surface condition of the land where the building was built. Possible values: n, o, t                |
| foundation_type                             | categorical | Type of foundation used while building. Possible values: h, i, r, u, w                             |
| roof_type                                   | categorical | Type of roof used while building. Possible values: n, q, x                                         |
| ground_floor_type                           | categorical | Type of the ground floor. Possible values: f, m, v, x, z                                           |
| other_floor_type                            | categorical | Type of constructions used in higher than the ground floors (except roof). Possible values: j, q, s, x |
| position                                    | categorical | Position of the building. Possible values: j, o, s, t                                               |
| plan_configuration                          | categorical | Building plan configuration. Possible values: a, c, d, f, m, n, o, q, s, u                       |
| has_superstructure_adobe_mud                | binary      | Flag variable indicating if the superstructure was made of Adobe/Mud                                 |
| has_superstructure_mud_mortar_stone         | binary      | Flag variable indicating if the superstructure was made of Mud Mortar - Stone                       |
| has_superstructure_stone_flag               | binary      | Flag variable indicating if the superstructure was made of Stone                                    |
| has_superstructure_cement_mortar_stone      | binary      | Flag variable indicating if the superstructure was made of Cement Mortar - Stone                    |
| has_superstructure_mud_mortar_brick         | binary      | Flag variable indicating if the superstructure was made of Mud Mortar - Brick                       |
| has_superstructure_cement_mortar_brick      | binary      | Flag variable indicating if the superstructure was made of Cement Mortar - Brick                    |
| has_superstructure_timber                   | binary      | Flag variable indicating if the superstructure was made of Timber                                   |
| has_superstructure_bamboo                   | binary      | Flag variable indicating if the superstructure was made of Bamboo                                   |
| has_superstructure_rc_non_engineered        | binary      | Flag variable indicating if the superstructure was made of non-engineered reinforced concrete        |
| has_superstructure_rc_engineered            | binary      | Flag variable indicating if the superstructure was made of engineered reinforced concrete            |
| has_superstructure_other                    | binary      | Flag variable indicating if the superstructure was made of any other material                       |
| legal_ownership_status                      | categorical | Legal ownership status of the land where the building was built. Possible values: a, r, v, w        |
| count_families                              | int         | Number of families that live in the building                                                          |
| has_secondary_use                           | binary      | Flag variable indicating if the building was used for any secondary purpose                         |
| has_secondary_use_agriculture               | binary      | Flag variable indicating if the building was used for agricultural purposes                         |
| has_secondary_use_hotel                     | binary      | Flag variable indicating if the building was used as a hotel                                        |
| has_secondary_use_rental                    | binary      | Flag variable indicating if the building was used for rental purposes                               |
| has_secondary_use_institution               | binary      | Flag variable indicating if the building was used as a location of any institution                   |
| has_secondary_use_school                    | binary      | Flag variable indicating if the building was used as a school                                       |
| has_secondary_use_industry                  | binary      | Flag variable indicating if the building was used for industrial purposes                           |
| has_secondary_use_health_post               | binary      | Flag variable indicating if the building was used as a health post                                  |
| has_secondary_use_gov_office                | binary      | Flag variable indicating if the building was used as a government office                            |
| has_secondary_use_police                    | binary      | Flag variable indicating if the building was used as a police station                               |
| has_secondary_use_other                     | binary      | Flag variable indicating if the building was used for other purposes                                |


## Conclusion and Future Work
This notebooks details the strategy used to predict earthquake damage given housing features.

1. GBM Ensemble - The current approach uses three popular GBMs, including CatBoost, XGBoost, and LightGBM. While each performs reasonably well on its own, the ensemble prediction involving all three GBMs demonstrates superior predictive power, as shwon below. This improved performance is observed even though bulk of the models are produced from simple perturbation, which is crude but implementationally straightforward;

<img src=".\assets\f1_distribution.png" alt="F1-scores of Various Models" width="800"/>

2. Auto-encoder - The strategy detailed here also includes an auto-encoder to capture geographic details using other features as well as the label in the dataset. This approach appears to have improved CatBoost model performance significantly, at least compared to the vanilla approach, where high cardinal geographic features are simply target encoded.

As future work, I would like to explore using:
1. more advanced geo-encoding approaches, e.g. TorchGeo, to better extract geography related features.

2. ordinal regression approaches, which was briefly explored using OrdinalGBT but produced odd results. It is noted here that the regression variants of the GBMs considered here were also explored at an early stage, but produced inferior results compared to the classifier variants;

3. stacked ensemble approaches.
